clear 
set matsize 11000
set more off, permanently
cd "../DATA EXPLORER CONTEST/Preprocessed_Data/merged_data_(t-1).csv"
capture log using DATA1601.log, replace //Bat dau ghi nhan qua trinh xu ly, tạo 1 file log ghi lại kết quả chạy mô hình

import excel DATA1601.xlsx, sheet("Sheet1") firstrow clear //nhập dữ liệu ngoại lai, file excel, sheet1
sort Date //sắp xếp quan sát theo 1 biến nào đó

*****Tao bien rate of return
gen Rvni = ln(VNI/VNI[_n-1]) 
gen Rvic = ln(VIC/VIC[_n-1])
drop if Rvni ==. //Rf la Rate of return roi, khong can generate nua
gen Rif = Rvic - Rf
gen Rff = Rvni - Rf

*****Thống kê mô tả
sum Rvic Rvni Rf VIC VNI //Thống kê mô tả tổng hợp các biến 
sum Rvic Rvni Rf VIC VNI, detail // Thống kê mô tả chi tiết các biến

***hoặc

tabstat Rvic Rvni Rf VIC VNI, statistics(mean, count, max, min, range, sd, variance, cv, semean, skewness, kurtosis, median) //thống kê chi tiết theo từng chỉ số muốn quan sát

***** Kiểm định
regress Rif Rff
***Kiểm định mô hình
ovtest //kiểm định Ramsey RESET

tsset Date //Định dạng chuỗi thời gian cho dữ liệu
estat durbinalt //Kiểm định Durbin
bgodfrey // Kiểm định Breusch-Godfrey LM bậc 1
estat bgodfrey,lags(2) // Kiểm định Breusch-Godfrey LM bậc cao

vif // Kiểm định đa cộng tuyến, tính hệ số p

dfuller Rff , lags(0) // Kiểm định tính dừng, chúng ta cần Test Statistics > Critial Value 
estat imtest // Kiểm định White, phương sai sai số thay đổi
estat hettest // Kiểm định Breusch-Godfrey, phương sai sai số thay đổi
***Với ví dụ này, mô hình bị phương sai thay đổi, không bị tự tương quan, chúng ta chỉ cần chạy Robust mà không cần VCE(robust)

***Chạy lại mô hình 
regress Rif Rff, robust //Không cần kiểm tra lại, mô hình mới sử dụng được 

eststo CAPM: regress Rif Rff, robust //triển khai mô hình hồi quy, thêm lệnh eststo: estimate store để dựng bảng
est table CAPM, ///
			stats(N r2 r2_a) b(%8.5f) stfmt(%8,5f) star(0.10 .05 .01) ///
			title(Table 1: CAPM)
			
save CAPM1601, replace

log close
exit			

