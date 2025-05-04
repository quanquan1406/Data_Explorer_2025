import streamlit as st
import pandas as pd
import requests
import os
import Token_use

st.set_page_config(page_title="💬 Chatbot Llama-Model", layout="wide")
st.title("🤖 Chatbot với model meta-llama/Llama-3-70B từ Together")

# --- Token API ---
HF_TOKEN = os.getenv("HF_TOKEN", Token_use.Token)
API_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# --- Gọi API ---
def query_llm(prompt):
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return f"⚠️ Lỗi: {response.status_code} - {response.text}"

# --- Tạo prompt từ toàn bộ file (giới hạn độ dài) ---
def format_prompt(df: pd.DataFrame, filename: str, max_chars: int = 3500):
    content = df.to_markdown(index=False)
    if len(content) > max_chars:
        content = content[:max_chars]
    prompt = f"Dữ liệu từ file **{filename}**:\n{content}\n\n"
    return prompt

# --- Upload 1 file duy nhất ---
st.sidebar.header("📂 Tải lên dữ liệu")
uploaded_file = st.sidebar.file_uploader(
    "Tải lên một file CSV hoặc Excel", type=["csv", "xlsx"], accept_multiple_files=False
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Không thể đọc file: {e}")
        df = None

    if df is not None:
        st.markdown(f"### 📄 File đã tải: **{uploaded_file.name}**")
        st.dataframe(df, use_container_width=True)

        user_input = st.chat_input("💬 Nhập câu hỏi về dữ liệu...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("🤖 Đang xử lý..."):
                    prompt = format_prompt(df, uploaded_file.name) + f"Câu hỏi: {user_input}"
                    answer = query_llm(prompt)
                    st.markdown(answer)
else:
    st.info("📌 Vui lòng tải lên một file để bắt đầu.")
