# 📊 Data Explorers 2025 – Dự báo giá cổ phiếu FPT 

## 🚀 Giới thiệu

Dự án được thực hiện trong khuôn khổ cuộc thi **Data Explorers 2025 - Vòng 2: Data-driven Business**, với mục tiêu xây dựng hệ thống dự báo giá cổ phiếu FPT (và CMG) dựa trên dữ liệu tài chính định lượng và phi cấu trúc. Nhóm kết hợp giữa mô hình học máy truyền thống và mô hình ngôn ngữ lớn (LLM) được tinh chỉnh bằng kỹ thuật **LoRA** để dự đoán **giá đóng cửa trong 3 ngày tiếp theo**.

## 🎯 Mục tiêu chính

1. Xây dựng mô hình học máy để dự báo giá cổ phiếu FPT dựa trên các đặc trưng kỹ thuật và tài chính.
2. Phát triển hệ thống RAG (Retrieval-Augmented Generation) cho phép truy vấn thông tin tài chính từ dữ liệu phi cấu trúc như tin tức và báo cáo.
3. Fine-tune mô hình LLM (LLaMA-2-7b-chat) sử dụng phương pháp QLoRA để dự báo giá cổ phiếu dựa trên dữ liệu thời gian và cảm xúc thị trường.

## 🧰 Công nghệ & thư viện

- **Ngôn ngữ**: Python 3.10+
- **Thư viện chính**: 
  - `transformers`, `peft`, `accelerate` – tinh chỉnh LLM
  - `scikit-learn`, `xgboost`, `lightgbm` – mô hình học máy truyền thống
  - `pandas`, `numpy` – xử lý dữ liệu
  - `matplotlib`, `seaborn` – trực quan hóa
  - `faiss`, `sentence-transformers` – hệ thống RAG
  - `streamlit` – giao diện demo


