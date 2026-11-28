# Bàn giao ngữ cảnh dự án cho AI khác

> Đọc file này trước khi sửa code. Đây là dự án do **một mình chủ dự án phụ trách**; không tiếp tục chia việc cho hai thành viên.

## Prompt tiếp tục công việc

Bạn đang tiếp quản dự án trong repository này. Hãy đọc `AI_HANDOFF.md`, `PROJECT_PLAN.md`, `README.md` và các báo cáo liên quan trước khi làm.

- Tiếp tục bằng tiếng Việt, giải thích ngắn gọn thay đổi và kết quả.
- Làm trên nhánh `develop`; ưu tiên thay đổi nhỏ, có thể chạy lại và cập nhật tài liệu/báo cáo đi cùng code.
- **Tập test đã được dùng một lần cho artifact Lasso alpha=0,001, 200 vòng. Không đọc, chấm, vẽ lại, chọn mô hình hoặc tinh chỉnh bằng tập test cũ. Không chạy `src.final_evaluation`.**
- Kết quả test trong `reports/10_final_test.md` chỉ áp dụng cho artifact alpha=0,001/200 vòng; không gán kết quả này cho bất kỳ cấu hình mới nào.
- Ứng viên CV/validation tốt hơn hiện có alpha=1e-5/1.000 vòng, nhưng relative step vẫn cao hơn tolerance, tiếp tục cải thiện ở biên 1.000 vòng, và chưa có holdout độc lập. Artifact/CLI chưa đổi.
- Không commit thư mục `data/`, dữ liệu Parquet, dự đoán từng tin hoặc thông tin nhạy cảm. `.gitignore` đã chặn dữ liệu và predictions row-level.
- Không thêm Docker hoặc giao diện web nếu chưa có nhu cầu triển khai cụ thể. CLI dự đoán đã có.
- Trước khi thay artifact bằng ứng viên mới, cần có dữ liệu thời gian mới và holdout tương lai được định nghĩa trước; không tái sử dụng test cũ.

## Mục tiêu dự án

Xây mô hình hồi quy **tự cài và tự huấn luyện** để dự đoán giá niêm yết nhà tại Việt Nam từ thông tin căn nhà, vị trí và thời điểm đăng tin. Phần dự đoán chính là NumPy FISTA Lasso; scikit-learn chỉ hỗ trợ tiền xử lý/metric và dùng làm nghiệm tham chiếu trong self-check nhỏ.

`price` là giá niêm yết trong tin đăng, **không phải giá giao dịch đã chốt**. `published_at` là ngày đăng. Không diễn giải kết quả như định giá chính thức hay dự báo chu kỳ dài.

## Dataset và phạm vi

- Nguồn: [`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates), Parquet, giấy phép CC BY-NC 4.0 (không dùng thương mại nếu chưa được cấp phép).
- Snapshot đã phân tích: 3.500.744 tin, 10 shard, khoảng 1,67 GB; thời gian 2025-06-01 đến 2026-03-30.
- Chỉ giữ `property_type_name == "Nhà"`, giá dương và ngày đăng hợp lệ: 1.493.988 tin.
- Giá lệch phải mạnh, có ngoại lai và giá rất nhỏ/lớn. Không xóa outlier âm thầm.
- Dataset local nằm trong `data/`, bị ignore bởi Git. `hf download` dùng theo README; dữ liệu chưa được commit lên GitHub.

## Quy trình đã hoàn thành

1. EDA trên 3,5 triệu tin; ghi missing, trùng, phân phối giá, theo thời gian và biểu đồ.
2. Tạo đặc trưng nhà và thời gian: `year`, `month`, `quarter`, `months_since_start`; không tạo giá/m² theo khu vực để tránh target leakage.
3. Chia theo ngày đăng, không shuffle: train 1.043.965 (2025-06-01–12-16), validation 226.346 (2025-12-17–2026-02-08), test 223.677 (2026-02-09–2026-03-30). Năm expanding-window fold chỉ nằm trong train.
4. Mọi imputer, percentile clipper, scaler và encoder chỉ fit trên train/fold-fit tương ứng. Số: median impute, clip P0.5–P99.5, standardize; category: most-frequent impute, one-hot với `min_frequency=25`.
5. Baseline median/NumPy OLS; TN1 Ridge/Lasso FISTA; TN2 NumPy MLP (50k mẫu tối đa/fold); TN3 NumPy autoencoder + Ridge head (50k/fold, bỏ district/ward).
6. So sánh theo metric, ổn định CV, khoảng cách fit-validation, residual và diễn giải; baseline và ba thử nghiệm ban đầu được ghi trong reports.
7. Đánh giá test **một lần** với Lasso alpha=0,001, giới hạn 200 vòng; fit trên train+validation.
8. Đã đóng gói artifact và CLI JSON cho một căn nhà; đã kiểm tra CLI mẫu.
9. Đã thử alpha/vòng lặp bổ sung bằng train/CV/validation; không dùng test.

## Kết quả cốt lõi và trạng thái model

### Artifact đang dùng

`models/vietnam_house_lasso.pkl` chứa preprocessor và hệ số Lasso FISTA, alpha=0,001, tối đa 200 vòng. CLI `src/predict_house.py` nạp đúng artifact này.

Validation của cấu hình này: RMSE/MAE log 0,5087/0,3605; 5-fold CV RMSE/MAE log 0,5288 ± 0,0155 / 0,3792 ± 0,0123. FISTA dừng ở giới hạn 200; relative step khoảng 0,00348, cao hơn tolerance 0,0001.

### Test đã dùng

Chỉ thuộc artifact alpha=0,001/200 vòng:

| Metric | Kết quả |
|---|---:|
| RMSE VND | 355,85 tỷ |
| MAE VND | 12,55 tỷ |
| MAPE | 2.045,20% |
| R² VND | 0,013 |
| RMSE log | 0,563 |
| MAE log | 0,396 |
| R² log | 0,630 |

MAPE rất cao do giá rất nhỏ trong test; metric giá gốc nhạy với outlier tới 129 nghìn tỷ. Residual cho thấy mô hình dự đoán cao ở nhóm giá thấp và thấp ở nhóm giá cao. Chi tiết chỉ trong `reports/10_final_test.md` và `reports/final_project_report.md`.

### Ứng viên nghiên cứu mới — chưa thay artifact

- Sweep alpha với 200 vòng: alpha 1e-5 tốt nhất trong lưới tinh chỉnh 1e-5–1e-3; CV RMSE/MAE log 0,4965 ± 0,0160 / 0,3557 ± 0,0111; validation 0,4775/0,3402.
- Đường học alpha 1e-5 tới 1.000 vòng: CV 0,4916 ± 0,0157 RMSE log, 0,3518 ± 0,0109 MAE log; validation 0,4722/0,3364. CV RMSE vẫn giảm tới checkpoint cuối.
- Relative step lớn nhất ở 1.000 vòng là 0,000308, vẫn cao hơn tolerance 0,0001.
- Alpha thấp nhất và vòng lặp cao nhất đều là biên lưới khảo sát. Kết quả có thể lạc quan vì alpha và số vòng đã được so sánh tuần tự trên cùng CV/validation. Đây chỉ là ứng viên nghiên cứu, chưa hội tụ và chưa được đánh giá bằng test độc lập.
- Không cập nhật artifact, CLI hay metric test. Muốn thay model cần dữ liệu mới, holdout thời gian mới và khóa lựa chọn trước khi chấm.

## Cấu trúc và tệp cần biết

- `src/features.py`: đặc trưng và pipeline chống leakage.
- `src/models/ridge_lasso.py`: NumPy FISTA Ridge/Lasso; mặc định alpha theo TN1, tối đa 200 vòng.
- `src/models/exp1_lasso_1000.py`: full CV/validation candidate 1.000 vòng.
- `src/models/lasso_alpha_sweep.py`: lưới alpha train/CV/validation.
- `src/models/lasso_iteration_sweep.py`: đường học theo vòng lặp và alpha.
- `src/models/check_optimizer.py`: self-check hệ số với sklearn trên dữ liệu tổng hợp.
- `src/final_evaluation.py`: **không chạy lại**; đã dùng để chấm test cũ.
- `src/predict_house.py`: CLI dự đoán từ JSON.
- `examples/house_input.json`: input mẫu.
- `reports/01_eda.md`–`reports/16_*.md`: EDA, quy trình, thử nghiệm, test và sweep mới.
- `reports/final_project_report.md`: báo cáo tổng hợp; `PROJECT_PLAN.md`: trạng thái/kế hoạch solo.
- `requirements.txt`: sklearn 1.8.0 được pin vì pickle; thư viện khác hiện chưa pin.

## Lệnh thường dùng

Từ repository root, PowerShell, sau khi cài dependencies và có dữ liệu:

```powershell
python -m src.eda
python -m src.prepare_houses
python -m src.split_data
python -m src.baseline
python -m src.models.ridge_lasso
python -m src.models.mlp_numpy
python -m src.models.hybrid_autoencoder
python -m src.analyze_validation_residuals
python -m src.compare_models
python -m src.models.check_optimizer
python -m src.predict_house examples/house_input.json
```

Các sweep alpha/iteration mỗi lệnh mất vài phút; lệnh và kết quả đầy đủ ghi trong `README.md`/`reports/14...16...`. Không chạy `python -m src.final_evaluation` để không đánh giá lại test.

## Môi trường và kiểm tra đã chạy

- Python 3.12; `.venv` được tạo sạch và cài từ `requirements.txt`.
- Scikit-learn 1.8.0 khớp pickle; CLI mẫu chạy không còn cảnh báo version.
- Trong `.venv` đã chạy lại EDA → prepare → split → baseline → TN1 → MLP → hybrid → residual → compare; không chạy final test.
- `python -m src.models.check_optimizer` đạt đối chiếu Ridge/Lasso trên dữ liệu tổng hợp.
- Các lưới alpha/iteration mới chạy thành công trên train/CV/validation.
- EDA từng báo không ghi được Matplotlib font cache ở user profile sandbox, nhưng vẫn tạo đủ report/biểu đồ; residual script chạy với `MPLCONFIGDIR` ở temp.
- Không chạy lại full download từ Hugging Face trong môi trường sạch; dữ liệu local sẵn có đã được dùng để chạy pipeline.

## Git snapshot

- Nhánh đang dùng: `develop`.
- Commit gần nhất tại lúc tạo file này: `8686f6f` (`Explore Lasso alpha and early stopping on temporal CV`), đã push lên `origin/develop`.
- `main` chưa được cập nhật từ commit khởi tạo; không tự merge/push main.
- Working tree đang sạch trước khi thêm file này.

## Việc hợp lý tiếp theo

Đừng tiếp tục tìm alpha/vòng lặp trên cùng test/CV mãi. Thu thập một snapshot dataset mới có ngày sau 2026-03-30, kiểm tra schema/license và thiết kế holdout tương lai **trước** khi tinh chỉnh. Sau đó: huấn luyện/chọn trên lịch sử train/CV/validation, khóa candidate, chấm holdout mới đúng một lần. Nếu chưa có dữ liệu mới, giữ artifact alpha=0,001/200 và báo rõ candidate 1e-5/1000 chỉ có CV/validation.

