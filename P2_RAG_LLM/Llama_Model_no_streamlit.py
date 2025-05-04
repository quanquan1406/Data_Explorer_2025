import pandas as pd
import requests
import os
import Token_use

# --- Cấu hình API ---
HF_TOKEN = os.getenv("HF_TOKEN", Token_use.Token)
API_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# --- Đọc dữ liệu từ file ---
FILE_PATH = "DATA EXPLORER CONTEST/Preprocessed_Data/merged_data_(t-1).csv"
try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Đã đọc dữ liệu từ {FILE_PATH}")
except Exception as e:
    print(f"❌ Không đọc được file: {e}")
    exit()

# --- Câu hỏi ---
QUESTION = "Tìm ra ngày mà Closing Price đạt giá cao nhất"

# --- Tạo prompt ---
def create_prompt(df, question, max_chars=3500):
    table_str = df.head(30).to_markdown(index=False)  # Giới hạn số dòng nếu cần
    prompt = f"Dưới đây là bảng dữ liệu:\n\n{table_str}\n\nCâu hỏi: {question}"
    return prompt[:max_chars]

# --- Gửi yêu cầu đến LLaMA ---
def query_llama(prompt):
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"⚠️ Lỗi {response.status_code}: {response.text}"

# --- Chạy ---
if __name__ == "__main__":
    prompt = create_prompt(df, QUESTION)
    print("📝 Prompt được gửi:")
    print(prompt)
    print("\n🤖 Đang gọi mô hình LLaMA...")
    result = query_llama(prompt)
    print("\n📤 Phản hồi từ LLaMA:")
    print(result)