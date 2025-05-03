* Load data
import delimited "../DATA EXPLORER CONTEST/Preprocessed_Data/merged_data.csv", clear

* Summary statistics
summarize

* Tính hệ số tương quan với closingprice
correlate closingprice totalvolume totalvalue marketcap pricechange matchedvolume matchedvalue sentiment_score foreigninvestorroom remainingroom matchedbuyvolume matchedbuyvalue matchedsellvolume matchedsellvalue negotiatedbuyvolume negotiatedbuyvalue negotiatedsellvolume negotiatedsellvalue totalbuyvolume totalbuyvalue totalsellvolume totalsellvalue eps loinhuan gt_khop kl_khop kl_ban kl_mua

* Vẽ biểu đồ ma trận để quan sát trực quan
graph matrix totalvolume totalvalue marketcap pricechange matchedvolume matchedvalue sentiment_score foreigninvestorroom remainingroom matchedbuyvolume matchedbuyvalue matchedsellvolume matchedsellvalue negotiatedbuyvolume negotiatedbuyvalue negotiatedsellvolume negotiatedsellvalue totalbuyvolume totalbuyvalue totalsellvolume totalsellvalue eps loinhuan gt_khop kl_khop kl_ban kl_mua closingprice, half

* Hồi quy tuyến tính với các biến có tương quan cao (ví dụ marketcap, loinhuan, eps)
regress closingprice marketcap loinhuan eps

