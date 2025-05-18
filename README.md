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

### 🔍 Mô hình Hồi quy tuyến tính

Mô hình hồi quy tuyến tính được lựa chọn là mô hình baseline trong quá trình thử nghiệm do tính đơn giản, dễ giải thích và hiệu quả trên tập dữ liệu nhỏ. Dữ liệu đầu vào bao gồm các đặc trưng kỹ thuật như:

- Giá đóng cửa phiên trước (`Closing Price_lag1`)
- Đường trung bình động (`MA5`, `MA10`)
- Khối lượng và giá trị giao dịch của nhà đầu tư nước ngoài
- Các chỉ số tài chính và sentiment từ tin tức

Sau khi huấn luyện trên tập dữ liệu gồm ~240 phiên giao dịch, mô hình đạt:

- **MSE**: 4.38 triệu
- **RMSE**: ~2,093 VND
- **MAE**: ~1,540 VND
- **R²**: 0.77
- **MAPE**: 1.06 % (Tỷ lệ sai số trung bình so với thực tế khoảng 1.06% )

![image](https://github.com/user-attachments/assets/66ab7706-4301-4bfd-adb3-ed2022dd207e)


