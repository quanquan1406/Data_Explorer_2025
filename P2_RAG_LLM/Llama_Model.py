import streamlit as st
import pandas as pd
import os
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import Token_use

# --- Cấu hình Together AI ---
HF_TOKEN = os.getenv("HF_TOKEN", Token_use.Token)
API_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# --- Chunk dataframe ---
def chunk_dataframe(df, chunk_size=5):
    chunks = []
    for i in range(0, len(df), chunk_size):
        sub_df = df.iloc[i:i+chunk_size]
        chunks.append(sub_df.to_csv(index=False))
    return chunks

# --- Load tất cả file từ thư mục ---
def load_all_data(folder_path):
    base = Path(folder_path)
    all_chunks = []
    for file_path in base.rglob("*.*"):
        if file_path.suffix.lower() in [".csv", ".xlsx"]:
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                chunks = chunk_dataframe(df)
                all_chunks.extend(chunks)
            except Exception as e:
                st.warning(f"⚠️ Không đọc được {file_path.name}: {e}")
    return all_chunks

# --- Embedding chunks ---
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    return embeddings, chunks, model

# --- FAISS index ---
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Truy vấn FAISS ---
def retrieve_top_chunks(index, chunks, query, model, k=5):
    query_vector = model.encode([query])
    _, I = index.search(query_vector, k=k)
    return [chunks[i] for i in I[0]]

# --- Gửi đến LLM ---
def query_llm(context, question):
    prompt = f"Dưới đây là thông tin có liên quan:\n{context}\n\nCâu hỏi: {question}"
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return f"❌ Lỗi {response.status_code}: {response.text}"

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📊 Chatbot RAG với dữ liệu bảng và Together.ai")

# --- Tự động load dữ liệu khi app khởi chạy ---
DEFAULT_FOLDER = "../DATA EXPLORER CONTEST"

if "rag_ready" not in st.session_state:
    with st.spinner(f"🚀 Đang tải dữ liệu từ thư mục: {DEFAULT_FOLDER}"):
        all_chunks = load_all_data(DEFAULT_FOLDER)
        if not all_chunks:
            st.error("❌ Không tìm thấy dữ liệu hợp lệ.")
        else:
            embeddings, chunk_texts, model = embed_chunks(all_chunks)
            index = create_faiss_index(np.array(embeddings))
            st.session_state["rag_ready"] = True
            st.session_state["index"] = index
            st.session_state["chunks"] = chunk_texts
            st.session_state["model"] = model
            st.success(f"✅ Đã xử lý {len(chunk_texts)} đoạn dữ liệu")

# --- Chat ---
if st.session_state.get("rag_ready"):
    user_question = st.chat_input("💬 Nhập câu hỏi của bạn về dữ liệu")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang truy vấn và tổng hợp thông tin..."):
                top_chunks = retrieve_top_chunks(
                    st.session_state["index"],
                    st.session_state["chunks"],
                    user_question,
                    st.session_state["model"]
                )
                context = "\n---\n".join(top_chunks)
                response = query_llm(context, user_question)
                st.markdown(response)
else:
    st.info("📌 Không có dữ liệu để truy vấn.")
