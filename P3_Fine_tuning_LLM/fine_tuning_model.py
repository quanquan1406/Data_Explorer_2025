import pandas as pd
import json
import os
import requests
from pathlib import Path
import Token_use

# --- Cấu hình ---
TOGETHER_API_KEY = Token_use.Token  # Nhập API Key của bạn
DATA_FOLDER = "./DATA EXPLORER CONTEST/News - FPT & CMG"  # Thư mục chứa file Excel
JSONL_PATH = "dataset.jsonl"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
N_EPOCHS = 3

# --- Bước 1: Chuyển Excel thành JSONL ---
def excel_to_jsonl(folder_path, output_path):
    all_files = Path(folder_path).rglob("*.xlsx")
    with open(output_path, "w", encoding="utf-8") as f:
        for file_path in all_files:
            try:
                df = pd.read_excel(file_path)
                # Cố gắng phát hiện xem file là kiểu giá cổ phiếu hay tin tức
                is_price_data = {"Mã cổ phiếu", "Ngày", "Giá đóng cửa", "Giá dự báo"}.issubset(df.columns)
                is_news_data = {"title", "date", "summary"}.issubset(df.columns)

                if is_price_data:
                    for _, row in df.iterrows():
                        try:
                            stock = str(row.get("Mã cổ phiếu", "Không xác định"))
                            date = str(row.get("Ngày", "Không xác định"))
                            price = str(row.get("Giá đóng cửa", "Không xác định"))
                            news = str(row.get("Tin tức", "Không có"))
                            prediction = str(row.get("Giá dự báo", "Không xác định"))

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
                            print(f"[!] Lỗi dòng trong file {file_path.name}: {e}")

                elif is_news_data:
                    for _, row in df.iterrows():
                        try:
                            title = str(row["title"])
                            date = str(row["date"])
                            summary = str(row["summary"])

                            prompt = (
                                f"Ngày: {date}\n"
                                f"Tiêu đề: {title}\n"
                                f"Nội dung: {summary}\n"
                                f"Hãy phân tích ảnh hưởng của thông tin này đến thị trường chứng khoán Việt Nam và các cổ phiếu liên quan."
                            )

                            assistant_reply = (
                                "Thông tin này có thể ảnh hưởng đến tâm lý nhà đầu tư và xu hướng giao dịch "
                                "của các cổ phiếu liên quan, đặc biệt là những cổ phiếu được đề cập trực tiếp trong tin."
                            )

                            item = {
                                "messages": [
                                    {"role": "system", "content": "Bạn là chuyên gia tài chính, phân tích tác động của tin tức đến thị trường."},
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": assistant_reply}
                                ]
                            }
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"[!] Lỗi dòng trong file {file_path.name}: {e}")
                else:
                    print(f"[!] File {file_path.name} không có cấu trúc hợp lệ. Bỏ qua.")

            except Exception as e:
                print(f"[!] Lỗi khi đọc file {file_path.name}: {e}")
    print(f"[✓] Đã tạo file JSONL: {output_path}")

# --- Bước 2: Upload file JSONL ---
def upload_file(jsonl_path):
    url = "https://api.together.xyz/v1/fine_tunes/files"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    try:
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
    except Exception as e:
        print(f"[✗] Lỗi khi upload file: {e}")
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
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            fine_tune_id = response.json().get("id")
            print(f"[✓] Đã bắt đầu fine-tune. ID: {fine_tune_id}")
            return fine_tune_id
        else:
            print(f"[✗] Fine-tune lỗi: {response.text}")
            return None
    except Exception as e:
        print(f"[✗] Lỗi khi gửi yêu cầu fine-tune: {e}")
        return None

# --- Gọi tất cả ---
def main():
    excel_to_jsonl(DATA_FOLDER, JSONL_PATH)
    file_id = upload_file(JSONL_PATH)
    if file_id:
        fine_tune_model(file_id)

if __name__ == "__main__":
    main()