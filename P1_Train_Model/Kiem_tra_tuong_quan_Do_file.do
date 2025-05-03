* Load data (giả sử đã lưu với các biến viết thường)
import delimited "../DATA EXPLORER CONTEST/Preprocessed_Data/merged_data_(t-1).csv", clear

* Summary statistics
summarize

correlate closing_price_lag1 *


