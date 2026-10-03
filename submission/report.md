# LightGBM Benchmark Report

Họ và tên: Nguyễn Huy Hoàng
Mã học viên: 2A202602738

1. Tôi sử dụng Google Cloud Platform (GCP), region `us-central1`, zone `us-central1-a`, với VM `e2-medium`.

2. Dataset Credit Card Fraud có 284,807 dòng và 31 cột, trong đó có 492 mẫu fraud. Dữ liệu được chia thành 60% train (170,883 mẫu), 20% validation (56,962 mẫu) và 20% test (56,962 mẫu), với seed = 16.

3. Thời gian load dữ liệu là 2.7681 giây; thời gian training LightGBM là 4.3136 giây; early stopping chọn best iteration = 68.

4. Trên tập test, mô hình đạt AUC-ROC = 0.97685, Accuracy = 0.99951, F1 = 0.84783, Precision = 0.90698 và Recall = 0.79592.

5. Inference latency cho 1 dòng là 1.2982 ms; throughput với batch 1,000 dòng là 264,668.60 dòng/giây. Thời gian được lấy theo median sau warm-up, sử dụng `predict_proba()` trên Pandas input.

6. Sau khi benchmark, VM có khoảng 3.8 GiB RAM, trong đó khoảng 503 MiB đang được sử dụng; tại thời điểm quan sát CPU gần như idle (99.8% idle). Network được kiểm tra bằng `ip -s link`. Ảnh CPU, RAM và Network được lưu trong `screenshots/`.

7. Billing chưa được ghi nhận Kết quả Billing được lưu trong ảnh tại `screenshots/`.

8. Tôi đã tải `benchmark.py` và `benchmark_result.json` từ VM về máy local và xóa các tài nguyên GCP của bài lab bằng `terraform destroy`. Bằng chứng cleanup: `[screenshot/delete_resource.png]`.