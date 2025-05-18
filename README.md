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

### 🧠 Hệ thống RAG – Retrieval-Augmented Generation

Bên cạnh mô hình dự báo định lượng, nhóm phát triển một hệ thống RAG (Retrieval-Augmented Generation) để **truy xuất thông tin tài chính và trả lời câu hỏi ngôn ngữ tự nhiên** dựa trên dữ liệu phi cấu trúc (tin tức, báo cáo tài chính, họp cổ đông...).

#### 🧩 Cách hoạt động

Hệ thống gồm 3 bước chính:

1. **Tiền xử lý dữ liệu văn bản**:
   - Dữ liệu từ các file tin tức, báo cáo tài chính, và giao dịch được chia nhỏ (chunking) thành từng đoạn văn bản.
   - Mỗi đoạn được nhúng (embedding) thành vector bằng mô hình `all-MiniLM-L6-v2` từ thư viện `sentence-transformers`.

2. **Tạo chỉ mục tìm kiếm (Vector Store)**:
   - Sử dụng thư viện **FAISS** để tạo chỉ mục tìm kiếm nhanh trên không gian vector.
   - Lưu cache để tối ưu thời gian truy vấn.

3. **Truy vấn với LLM**:
   - Khi người dùng đặt câu hỏi, hệ thống:
     - Mã hóa câu hỏi → truy tìm các đoạn văn bản liên quan trong FAISS.
     - Tổng hợp context từ các đoạn tìm được → gửi vào mô hình **LLaMA-3.3-70B** thông qua API của **Together.ai**.
   - LLM sinh câu trả lời từ context, có thể kèm dẫn nguồn (sheet/row) từ dữ liệu gốc.

#### 🔧 Công cụ sử dụng

- `FAISS`: Tìm kiếm gần đúng theo vector hiệu quả.
- `sentence-transformers`: Tạo embedding từ văn bản.
- `Together.ai`: API truy cập mô hình ngôn ngữ lớn LLaMA 3.
- `Streamlit`: Triển khai giao diện demo hỏi–đáp.
- `pandas`, `numpy`: Phân tích dữ liệu bảng.

#### 💻 Hướng dẫn chạy demo RAG

```bash
cd P2_RAG_LLM
streamlit run Llama_model.py
```
![image](https://github.com/user-attachments/assets/a925142e-07d4-49cb-b26b-932a715f5946)

![image](https://github.com/user-attachments/assets/fba8a24f-e621-4873-a201-76f0fa5bd4e2)

![image](https://github.com/user-attachments/assets/d00387cc-5906-43a3-ac8c-e9b4b36b7bf2)



