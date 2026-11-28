# Kế hoạch dự án — một người phụ trách

Chủ dự án tự thực hiện và chịu trách nhiệm toàn bộ repository. Các mục chia theo hai thành viên trước đây đã được gộp và bỏ.

## Trạng thái hiện tại

- Hoàn thành EDA, tạo đặc trưng, chia thời gian, baseline, ba thử nghiệm, so sánh, đánh giá cuối và CLI dự đoán.
- Báo cáo tổng kết: `reports/final_project_report.md`.
- Mô hình/pipeline hiện hành: `models/vietnam_house_lasso.pkl` (Lasso FISTA tối đa 200 vòng).
- Tập test đã được dùng một lần cho báo cáo hiện tại; không dùng lại để chọn mô hình hoặc siêu tham số.
- Kiểm tra optimizer và ứng viên 1.000 vòng chỉ sử dụng dữ liệu tổng hợp/train/CV/validation; artifact cuối vẫn là cấu hình 200 vòng.
- Lưới alpha/đường học mới nhất tìm ứng viên alpha=0,00001/1.000 vòng (CV RMSE log 0,4916 ± 0,0157; validation 0,4722), nhưng còn cải thiện ở biên 1.000 vòng, relative step cao hơn tolerance và chưa có test độc lập. CLI/artifact chưa đổi.

## Việc cần làm nếu tiếp tục cải thiện mô hình

1. Không mở rộng tinh chỉnh trên cùng holdout mãi; ứng viên alpha thấp cần được khóa bằng quy trình CV định trước và kiểm chứng trên dữ liệu thời gian mới.
2. Ghi metric cả thang log và VND, độ ổn định theo fold, thời gian huấn luyện, phân tích lỗi theo nhóm giá/thời gian và giới hạn dữ liệu.
3. Không gán kết quả test cũ cho cấu hình mới. Muốn có đánh giá cuối độc lập cho mô hình mới, cần thu thập dữ liệu mới theo thời gian và thiết kế holdout mới trước khi thử nghiệm.
4. Giữ nguyên dữ liệu gốc ngoài Git; không đưa tin đăng, nhãn hay dự đoán từng dòng lên GitHub.

## Đóng gói/triển khai

CLI một căn nhà đã có tại `src/predict_house.py`. Docker và giao diện web chưa cần cho dự án hiện tại; chỉ thêm khi có yêu cầu triển khai hoặc chạy đồng nhất trên CI.
