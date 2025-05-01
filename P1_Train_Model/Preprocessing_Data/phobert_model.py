import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

# Đọc dữ liệu từ file Excel
df = pd.read_excel("../../DATA EXPLORER CONTEST/News - FPT & CMG/CafeF_News_FPT_CMG.xlsx")

# Ghép title 
df['input_text'] = df['title'].astype(str)

# PhoBERT đã fine-tune cho phân tích cảm xúc
model_name = "wonrax/phobert-base-vietnamese-sentiment"

# Load model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tạo pipeline cho phân tích cảm xúc
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Hàm tách từ thủ công
def custom_tokenize(text):
    # Tách từ dựa trên khoảng trắng và dấu câu
    text = text.strip()  # Loại bỏ khoảng trắng đầu và cuối
    words = re.findall(r'\w+|[^\w\s]', text)  # Tìm tất cả các từ và dấu câu
    return words

# Hàm phân tích cảm xúc với tách từ thủ công
def analyze_sentiment(text):
    try:
        # Tách từ thủ công
        segmented_text = custom_tokenize(text)
        # Chuyển lại thành chuỗi để gửi vào pipeline
        segmented_text = " ".join(segmented_text)
        result = sentiment_pipeline(segmented_text[:512])[0]
        label = result['label']
        if label == 'POS':
            return 'positive'
        elif label == 'NEU':
            return 'neutral'
        elif label == 'NEG':
            return 'negative'
        return 'neutral'
    except Exception as e:
        print(f"Error: {e} | Text: {text[:50]}")
        return 'neutral'

# Áp dụng phân tích cảm xúc vào dữ liệu
df['sentiment'] = df['input_text'].apply(analyze_sentiment)

# Lưu kết quả ra file Excel
df.to_excel("../../DATA EXPLORER CONTEST/Preprocessed_Data/output_with_sentiment.xlsx", index=False)

# In một vài kết quả để kiểm tra
print(df[['title', 'date', 'summary', 'sentiment']].head())
