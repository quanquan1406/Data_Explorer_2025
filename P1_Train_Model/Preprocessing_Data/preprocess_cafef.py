import pandas as pd

# Đọc file đã có sentiment và sentiment_score
df = pd.read_excel("../../DATA EXPLORER CONTEST/Preprocessed_Data/output_with_sentiment_score.xlsx")

# Đảm bảo cột ngày ở đúng định dạng
df['date'] = pd.to_datetime(df['date']).dt.date

# Hàm lấy sentiment phổ biến nhất trong mỗi ngày
def get_dominant_sentiment(group):
    # Đếm tần suất sentiment
    sentiment_counts = group['sentiment'].value_counts()
    top_sentiment = sentiment_counts.idxmax()
    
    # Lọc ra hàng đầu tiên có sentiment thống trị
    representative_row = group[group['sentiment'] == top_sentiment].iloc[0]

    return pd.Series({
        'date': representative_row['date'],
        'sentiment': representative_row['sentiment'],
        'sentiment_score': representative_row['sentiment_score'],
    })

# Áp dụng theo từng nhóm ngày
result_df = df.groupby('date').apply(get_dominant_sentiment).reset_index(drop=True)

# Xuất ra file
result_df.to_excel("../../DATA EXPLORER CONTEST/Preprocessed_Data/daily_dominant_sentiment_no_title.xlsx", index=False)

# In kiểm tra vài dòng
print(result_df.head())
