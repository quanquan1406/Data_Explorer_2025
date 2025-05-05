import streamlit as st
import pandas as pd
import os
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import Token_use
import pickle

# --- Cấu hình Together AI ---
HF_TOKEN = os.getenv("HF_TOKEN", Token_use.Token)
API_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# --- Chunk row-level ---
def chunk_dataframe_rows(df, file_name, sheet_name=""):
    chunks = []
    for row_idx in df.index:
        row_data = df.loc[row_idx].astype(str)
        if row_data.isnull().all() or row_data.str.strip().eq("nan").all():
            continue
        row_text = " | ".join([f"{col}: {val}" for col, val in row_data.items()])
        tag = f"[{file_name} | Sheet: {sheet_name} | Row: {row_idx+1}]"
        chunks.append(f"{tag}\n{row_text}")
    return chunks

# --- Cache mô hình ---
@st.cache_resource(show_spinner="🔄 Đang tải SentenceTransformer...")
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# --- Đường dẫn cache ---
CACHE_FOLDER = "./cache"
Path(CACHE_FOLDER).mkdir(parents=True, exist_ok=True)
INDEX_FILE = os.path.join(CACHE_FOLDER, "faiss.index")
CHUNKS_FILE = os.path.join(CACHE_FOLDER, "chunks.pkl")
EMBEDDINGS_FILE = os.path.join(CACHE_FOLDER, "embeddings.npz")

# --- Load và xử lý dữ liệu hoặc lấy từ cache ---
@st.cache_resource(show_spinner="📦 Đang xử lý dữ liệu...")
def process_data_and_build_index(folder_path):
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE):
        st.info("🧠 Đang tải lại model & dữ liệu từ cache")
        model = load_model()
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        return index, all_chunks, model

    base = Path(folder_path)
    all_chunks = []

    for file_path in base.rglob("*.*"):
        if file_path.suffix.lower() not in [".csv", ".xlsx"]:
            continue
        file_name = file_path.name
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
                chunks = chunk_dataframe_rows(df, file_name)
                all_chunks.extend(chunks)
            else:
                xl = pd.ExcelFile(file_path, engine="openpyxl")
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    if df.empty:
                        continue
                    chunks = chunk_dataframe_rows(df, file_name, sheet_name)
                    all_chunks.extend(chunks)
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi đọc file {file_name}: {e}")

    if not all_chunks:
        return None, None, None

    model = load_model()

    # Batch embedding
    embeddings = []
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(batch_embeddings)
    all_embeddings = np.vstack(embeddings)

    np.savez_compressed(EMBEDDINGS_FILE, embeddings=all_embeddings)

    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)
    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    return index, all_chunks, model

# --- Truy vấn FAISS ---
def retrieve_top_chunks(index, chunks, query, model, k=5):
    query_vector = model.encode([query], normalize_embeddings=True)
    _, I = index.search(query_vector, k=k)
    return [chunks[i] for i in I[0]]

# --- Gửi đến LLM ---
def query_llm(context, question):
    prompt = f"""
Dưới đây là các dòng dữ liệu có thể liên quan:

{context}

Câu hỏi: {question}
Vui lòng trả lời chính xác, rõ ràng, và nếu có thể, hãy chỉ ra nguồn từ dòng nào.
"""
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return f"❌ Lỗi {response.status_code}: {response.text}"

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📊 Chatbot RAG – Truy xuất theo hàng từ dữ liệu lớn")

if st.sidebar.button("🧹 Xóa cache và xử lý lại dữ liệu"):
    for f in [INDEX_FILE, CHUNKS_FILE, EMBEDDINGS_FILE]:
        if os.path.exists(f):
            os.remove(f)
    st.experimental_rerun()

DEFAULT_FOLDER = "../DATA EXPLORER CONTEST"
index, chunk_texts, model = process_data_and_build_index(DEFAULT_FOLDER)

if index is None:
    st.error("❌ Không tìm thấy dữ liệu hợp lệ trong thư mục.")
else:
    st.success(f"✅ Đã xử lý {len(chunk_texts)} dòng dữ liệu")
    user_question = st.chat_input("💬 Nhập câu hỏi của bạn về dữ liệu")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang truy vấn và tổng hợp thông tin..."):
                top_chunks = retrieve_top_chunks(index, chunk_texts, user_question, model)
                context = "\n---\n".join(top_chunks)
                response = query_llm(context, user_question)
                st.markdown(response)
