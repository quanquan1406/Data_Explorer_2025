import pandas as pd
import json
import os
import requests
import Token_use

# --- Cấu hình ---
TOGETHER_API_KEY = Token_use.Token  # Thay bằng khóa API thật
EXCEL_PATH = "./DATA EXPLORER CONTEST/News - FPT & CMG/CafeF_News_FPT_CMG.xlsx"  # Tên file Excel
JSONL_PATH = "dataset.jsonl"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Hoặc model khác từ Together.ai
N_EPOCHS = 3

# --- Bước 1: Chuyển đổi Excel → JSONL ---
def excel_to_jsonl(excel_path, output_path):
    df = pd.read_excel(excel_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            try:
                stock = str(row["Mã cổ phiếu"])
                date = str(row["Ngày"])
                price = str(row["Giá đóng cửa"])
                news = str(row.get("Tin tức", "Không có"))
                prediction = str(row["Giá dự báo"])  # VD: "96.8, 97.2, 97.9"

                prompt = (
                    f"Cổ phiếu: {stock}\n"
                    f"Ngày: {date}\n"
                    f"Giá hiện tại: {price}\n"
                    f"Tin tức: '{news}'.\n"
                    f"Dự báo giá 3 ngày tới."
                )

                item = {
                    "messages": [
                        {"role": "system", "content": "Bạn là chuyên gia tài chính."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": prediction}
                    ]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[!] Lỗi dòng: {e}")

    print(f"[✓] Đã tạo file JSONL: {output_path}")


# --- Bước 2: Upload file JSONL lên Together ---
def upload_file(jsonl_path):
    url = "https://api.together.xyz/v1/fine_tunes/files"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    }
    with open(jsonl_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        file_id = response.json().get("id")
        print(f"[✓] Đã upload file. ID: {file_id}")
        return file_id
    else:
        print(f"[✗] Upload lỗi: {response.text}")
        return None


# --- Bước 3: Gửi yêu cầu fine-tune ---
def fine_tune_model(file_id):
    url = "https://api.together.xyz/v1/fine_tunes"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "training_file": file_id,
        "model": MODEL_NAME,
        "n_epochs": N_EPOCHS
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        fine_tune_id = response.json().get("id")
        print(f"[✓] Đã bắt đầu fine-tune. ID: {fine_tune_id}")
        return fine_tune_id
    else:
        print(f"[✗] Fine-tune lỗi: {response.text}")
        return None


# --- Gọi tất cả ---
def main():
    excel_to_jsonl(EXCEL_PATH, JSONL_PATH)
    file_id = upload_file(JSONL_PATH)
    if file_id:
        fine_tune_model(file_id)

if __name__ == "__main__":
    main()
