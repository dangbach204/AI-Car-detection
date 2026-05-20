# Car Detection (HOG + SVM)

Project phát hiện xe (car detection) sử dụng đặc trưng HOG (Histogram of Oriented Gradients)
và mô hình phân loại SVM (Support Vector Machine). Ứng dụng bao gồm pipeline thu hình,
phát hiện đối tượng thời gian thực và giao diện web đơn giản để xem kết quả.

## Tính năng

- Phát hiện xe bằng HOG + SVM
- Xử lý video / webcam theo luồng (real-time)
- Giao diện web hiển thị video với khung nhận diện

## Yêu cầu

- Python 3.8+
- Các thư viện (xem `requirements.txt`): `flask`, `opencv-python`, `numpy`, `scipy`, `scikit-learn==1.7.2`, `joblib`

## Cài đặt

1. Tạo virtual environment và kích hoạt (Windows):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. (Tùy chọn) Nếu dùng Linux / macOS, dùng `source venv/bin/activate` thay cho lệnh trên.

## Chạy ứng dụng

1. Từ thư mục gốc của project:

```powershell
python app/app.py
```

2. Mở trình duyệt vào: http://localhost:5000

Ứng dụng sẽ tự khởi tạo các luồng camera và luồng phát hiện (xem `app/app.py`).

## Cấu trúc dự án (tóm tắt)

- `app/` — mã nguồn web và các thành phần runtime
  - `app.py` — entrypoint chạy server và khởi tạo luồng
  - `camera.py` — xử lý luồng video / capture
  - `detector.py` — logic trích xuất HOG và gọi mô hình SVM để dự đoán
  - `routes.py` — định nghĩa route Flask
  - `state.py`, `config.py` — trạng thái và cấu hình chung
- `models/` — nơi lưu mô hình đã huấn luyện (ví dụ: `svm_hog.joblib`)
- `static/`, `templates/` — tài nguyên web
- `requirements.txt` — danh sách phụ thuộc

## Mô hình và huấn luyện

- Dự án dùng HOG để trích đặc trưng ảnh và SVM (scikit-learn) để huấn luyện bộ phân loại.
- Nếu bạn muốn huấn luyện lại mô hình: thu tập dữ liệu ảnh có nhãn (xe / không xe),
  trích HOG cho từng patch, huấn luyện `sklearn.svm.SVC` hoặc `LinearSVC`,
  và lưu mô hình bằng `joblib.dump()` vào thư mục `models/`.

## Dữ liệu

Do giới hạn kích thước, dataset không được đính kèm trong repo. Nếu bạn có dataset riêng,
đặt nó theo cấu trúc mà script huấn luyện của bạn yêu cầu, sau đó huấn luyện và lưu mô hình
vào `models/`.

## Gợi ý mở rộng

- Thay SVM bằng mô hình CNN để cải thiện độ chính xác
- Thêm lọc theo khu vực (ROI) hoặc tracking để giảm nhiễu
- Tối ưu hóa tốc độ bằng cách giảm độ phân giải hoặc dùng batching

## License & Contact

Nêu rõ license nếu cần hoặc liên hệ tác giả để biết thêm chi tiết.

---

📥 Dataset (nếu có link tải):
[https://drive.google.com/drive/folders/1ahzPTBAhwJ6qJTrfJh71wTzvEqR78HfL?usp=sharing](https://drive.google.com/drive/folders/1ahzPTBAhwJ6qJTrfJh71wTzvEqR78HfL?usp=sharing)
