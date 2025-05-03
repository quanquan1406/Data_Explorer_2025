import streamlit as st
import pandas as pd
import requests
import os
from pathlib import Path

import Token_use

st.set_page_config(page_title="💬 Chatbot Llama-Model", layout="wide")
st.title("🤖 Chatbot với model meta-llama/Llama-Model từ Together")

# --- TOKEN Together.ai ---
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return f"⚠️ Lỗi: {response.status_code} - {response.text}"

# --- Load dữ liệu từ folder con recursively ---
@st.cache_data
def load_data(root_dir="data"):
    dfs = {}
    base = Path(root_dir)
    for file_path in base.rglob("*.*"):
        if file_path.suffix.lower() in [".csv", ".xlsx"]:
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                key = str(file_path.relative_to(base))
                dfs[key] = df
            except Exception as e:
                st.warning(f"Không đọc được {file_path}: {e}")
    return dfs

dfs = load_data("../DATA EXPLORER CONTEST")

# --- Sidebar: Navigation ---
st.sidebar.header("📚 Điều hướng")
page = st.sidebar.radio("Chọn trang:", ["Chatbot", "Lịch sử chat"])

# --- Chat history ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Hiển thị chat history ---
def display_chat():
    for turn in st.session_state.chat_history:
        st.chat_message(turn['role']).markdown(turn['content'])

# --- Format prompt giới hạn độ dài ---
def format_chat(chat_history, dfs, last_n=3, max_prompt_length=3000):
    prompt = ""
    for name, df in dfs.items():
        prompt += f"Dữ liệu từ {name}:\n{df.head(2).to_markdown(index=False)}\n\n"
        if len(prompt) > max_prompt_length:
            break
    recent = chat_history[-last_n*2:]
    for turn in recent:
        tag = 'User' if turn['role']=='user' else 'Assistant'
        prompt += f"{tag}: {turn['content']}\n"
    return prompt[:max_prompt_length]

# --- Trang Chatbot ---
if page == "Chatbot":
    st.subheader("💬 Giao tiếp với dữ liệu")

    st.sidebar.header("📊 Dữ liệu đã nạp:")
    for name, df in dfs.items():
        st.sidebar.markdown(f"**{name}**")
        st.sidebar.dataframe(df.head())

    user_input = st.chat_input("💬 Nhập câu hỏi về dữ liệu...")
    if user_input:
        st.session_state.chat_history.append({'role':'user','content':user_input})
        display_chat()
        prompt = format_chat(st.session_state.chat_history, dfs)
        with st.chat_message('assistant'):
            with st.spinner("🤖 Đang xử lý..."):
                answer = query_llm(prompt)
                st.markdown(answer)
        st.session_state.chat_history.append({'role':'assistant','content':answer})

    if st.session_state.chat_history:
        display_chat()

# --- Trang Lịch sử chat ---
elif page == "Lịch sử chat":
    st.subheader("🕘 Lịch sử trao đổi")
    if st.session_state.chat_history:
        display_chat()
    else:
        st.info("Chưa có lịch sử chat nào.")