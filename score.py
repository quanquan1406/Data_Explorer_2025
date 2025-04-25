import pandas as pd

# Đọc file Excel đã xử lý cảm xúc
df = pd.read_excel("output_with_sentiment.xlsx")

# Ánh xạ sentiment sang chỉ số
sentiment_mapping = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}

# Thêm cột sentiment_score
df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

# Lưu lại ra file mới nếu cần
df.to_excel("output_with_sentiment_score.xlsx", index=False)

# In kiểm tra vài dòng đầu
print(df[['title', 'sentiment', 'sentiment_score']].head())