# Bàn giao công việc cho nhóm

## Trạng thái hiện tại

Dự án đã có pipeline EDA, xử lý dữ liệu, chia thời gian, baseline, ba thử nghiệm, so sánh, đánh giá cuối và artifact Lasso. Mô hình thắng rubric hiện tại là Lasso NumPy tự cài; metric test và các giới hạn nằm trong `reports/10_final_test.md`.

Đây là bản hoàn thành phần nghiên cứu ban đầu, chưa phải mô hình production đã được xác nhận. Dữ liệu là tin đăng nhà ở Việt Nam, `price` là giá niêm yết và `published_at` là ngày đăng. Dataset chỉ bao phủ khoảng 10 tháng; không được diễn giải kết quả thành giá giao dịch hoặc dự báo dài hạn.

## Nguyên tắc chung — cả hai thành viên

1. Làm trên nhánh `develop`, mỗi người tạo nhánh riêng từ develop và mở pull request; không đẩy thẳng vào `main`.
2. Không commit `data/`, file Parquet/CSV dữ liệu, dự đoán từng dòng hoặc thông tin riêng tư. `.gitignore` đã được thêm để chặn các nội dung này.
3. Không mở lại/chấm/điều chỉnh theo `test.parquet`. Tập test đã được dùng cho đánh giá cuối. Mọi thay đổi mô hình phải dựa trên train, time-series CV và validation; nếu đổi mô hình thì ghi kết quả là thử nghiệm mới, không gọi đó là kết quả test độc lập mới.
4. Không thay hoặc xóa số liệu bất lợi. Ghi cấu hình, seed, lệnh, kết quả trước/sau và hạn chế.
5. Trước khi bắt đầu: `git switch develop; git pull origin develop; git switch -c task/<ten-ngan>`.
6. Khi hoàn tất: chạy các lệnh kiểm tra được giao, xem `git status`, commit phần việc của mình, push nhánh task và gửi PR có tóm tắt thay đổi/kết quả.

## Thành viên 1 — kiểm tra và cải thiện TN1 Lasso

**Mục tiêu:** xử lý rủi ro đã biết là FISTA dừng ở 200 vòng nhưng relative change khoảng 0,00346, cao hơn tolerance 0,0001; đồng thời xác minh mô hình tự cài và kết quả có thể tái lập.

**Việc cần làm:**

1. Đọc `src/models/ridge_lasso.py`, `src/features.py`, `reports/05_exp1_ridge_lasso.md` và `reports/09_model_comparison.md`.
2. Kiểm tra công thức gradient/proximal, Lipschitz constant, intercept, chuẩn hóa, penalty và điều kiện dừng. Viết kiểm tra nhỏ trên ma trận tổng hợp để đối chiếu nghiệm của Ridge/Lasso với nghiệm tham chiếu chỉ dùng làm kiểm chứng (không dùng mô hình sklearn làm mô hình chính).
3. Tìm nguyên nhân hội tụ chậm; thử thay đổi optimizer/tolerance/alpha chỉ trên train và time-series CV, lựa chọn cuối cùng bằng validation. Không dùng test để chọn.
4. Báo cáo metric fit/validation và mean ± std của 5 fold, số vòng lặp, relative change, thời gian chạy; so sánh công bằng với cấu hình hiện tại. Nếu chưa giải quyết được, giữ nguyên kết quả cũ và ghi rõ.
5. Cập nhật report, kết quả tổng hợp và bảng tracker nếu có thay đổi. Không chạy `src/final_evaluation.py` và không tạo báo cáo test mới.

**Bàn giao đạt yêu cầu khi:** PR có giải thích toán/cấu hình, kiểm tra nhỏ tái lập được, bảng CV + validation, kết luận có/không thay cấu hình Lasso hiện tại; không có file dữ liệu hoặc metric test mới.

## Thành viên 2 — rà soát dữ liệu, tài liệu và khả năng chạy lại

**Mục tiêu:** giúp người khác thiết lập dự án và tái chạy các bước mà không phụ thuộc vào thư mục dữ liệu cục bộ của người tạo.

**Việc cần làm:**

1. Đọc `README.md`, `PROJECT_TRACKER.md`, `src/prepare_houses.py`, `src/split_data.py`, `src/features.py` và báo cáo EDA/feature/split.
2. Viết hướng dẫn tải/đặt 10 shard dataset từ Hugging Face, nêu license CC BY-NC 4.0, schema cần có, phiên bản Python/thư viện và thứ tự lệnh từ EDA đến các thí nghiệm. Không đưa dataset lên GitHub.
3. Rà soát logic ngày đăng, lọc loại `Nhà`, giá dương, thứ tự thời gian, chia theo ngày và fit preprocessing riêng trong từng fold. Ghi bất kỳ vấn đề nào trước khi sửa; không sửa số liệu đầu vào để làm đẹp metric.
4. Kiểm tra các lệnh trong README trên môi trường sạch nếu có thể; sửa đường dẫn/lệnh sai và ghi kết quả. Bảo đảm `requirements.txt` phản ánh các thư viện thực sự dùng.
5. Cập nhật tracker cho đúng với kết quả hiện hành. Có thể đề xuất Docker, nhưng chỉ thêm khi chứng minh được nhu cầu cụ thể (ví dụ triển khai hoặc môi trường chạy giống nhau); Docker không thay thế kiểm tra dữ liệu/mô hình.

**Bàn giao đạt yêu cầu khi:** người mới có thể cài môi trường, lấy dữ liệu theo hướng dẫn, chạy pipeline không vướng lệnh/đường dẫn; báo cáo rà soát có checklist kết quả và mọi hạn chế còn lại. Không chấm lại test.

## Trạng thái tiếp tục (2026-11-28)

- Chưa thấy nhánh hoặc commit mới của thành viên nào trên remote `develop`.
- Đã bổ sung hướng dẫn tải dataset/chạy pipeline trong `README.md` và thêm `huggingface_hub` để có lệnh `hf download`.
- Đã chẩn đoán FISTA trên 30.000 dòng train; xem `reports/11_lasso_optimizer_diagnostic.md`. Đây chưa phải chạy đủ CV/validation và chưa thay artifact cuối.
- Thành viên 1 vẫn cần quyết định/chạy 1.000 vòng trên đủ train/CV/validation nếu muốn đề xuất cấu hình mới; không dùng test.
- Thành viên 2 vẫn cần kiểm tra hướng dẫn trên môi trường sạch; môi trường sạch chưa được dựng trong lần cập nhật này.

## Thứ tự phối hợp

- Làm song song trên hai nhánh riêng; Thành viên 1 chỉ sửa chủ yếu phần mô hình/report TN1, Thành viên 2 tập trung README/data pipeline/tracker.
- Nếu Thành viên 1 thay cấu hình thắng, báo Thành viên 2 để đồng bộ tài liệu; không ghi đè thay đổi của nhau.
- Chủ dự án xem hai PR, giải quyết xung đột, sau đó mới quyết định có cần chạy lại đánh giá cuối hay không. Vì test đã dùng, lần chạy lại chỉ được xem là kiểm tra hậu chỉnh sửa, không được dùng để lựa chọn/tinh chỉnh mô hình.

## Docker có cần không?

Hiện chưa cần Docker để hoàn thành bài tập hoặc để commit/push code. `requirements.txt` và hướng dẫn môi trường Python là đủ cho giai đoạn này. Docker chỉ nên bổ sung khi nhóm cần chạy giống nhau trên nhiều máy/CI hoặc triển khai demo; nếu thêm, image phải lấy dữ liệu bên ngoài, không đóng gói dataset 1,7 GB hay test predictions.
