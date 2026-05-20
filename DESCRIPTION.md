# Vehicle Detection & Counting System - Project Description

## Tổng Quan Dự Án

Đây là một **hệ thống phát hiện và đếm xe tự động** kết hợp giữa mô hình máy học cổ điển (HOG + SVM) và ứng dụng web thời thực. Dự án được thiết kế để xử lý luồng video từ camera với hiệu suất cao và độ chính xác tốt.

---

## Mô Hình AI - Kỹ Thuật & Training

### 1. **Kỹ Thuật Trích Xuất Đặc Trưng: HOG (Histogram of Oriented Gradients)**

HOG là kỹ thuật trích xuất đặc trưng hình ảnh cổ điển, rất hiệu quả cho phát hiện vật thể:

- **Nguyên lý**: Chia hình ảnh thành các ô nhỏ (cells) và tính toán histogram của hướng gradient (cạnh) trong mỗi ô
- **Tham số HOG được sử dụng**:
  - Window size: **64×64 pixels** (kích thước cửa sổ phát hiện)
  - Block size: **16×16 pixels**
  - Block stride: **8×8 pixels**
  - Cell size: **8×8 pixels**
  - Orientation bins: **9** (9 hướng)
  - **Kích thước vector đặc trưng**: 1764 features

- **Ưu điểm**:
  - Nhanh, hiệu quả về mặt tính toán
  - Bất biến với phép quay nhẹ và thay đổi ánh sáng
  - Hoạt động tốt với ảnh đen trắng/grayscale
  - Thích hợp cho phát hiện xe

### 2. **Thuật Toán Phân Loại: SVM (Support Vector Machine)**

SVM được sử dụng để phân loại patch hình ảnh thành 2 lớp:

- **Lớp Positive**: Chứa xe (car)
- **Lớp Negative**: Không chứa xe (background)

- **Mô hình**: `LinearSVC` từ scikit-learn
- **Pipeline**: StandardScaler (chuẩn hóa) → LinearSVC (phân loại)

### 3. **Quy Trình Training**

Các bước training mô hình:

1. **Chuẩn bị dữ liệu**:
   - Tập positive: Các mẫu 64×64 chứa xe
   - Tập negative: Các mẫu 64×64 không chứa xe
   - Chia: 80% training, 20% testing

2. **Trích xuất HOG features**:
   - Tính HOG cho toàn bộ dataset
   - Kích thước output: `(số_mẫu, 1764)`

3. **Chuẩn hóa dữ liệu** (StandardScaler):
   - Để các features có phân bố chuẩn (mean=0, std=1)
   - Cải thiện hiệu suất SVM

4. **Training SVM**:
   - Tìm siêu phẳng tối ưu để phân tách hai lớp
   - Sử dụng LinearSVC (nhanh, phù hợp với dữ liệu lớn)

5. **Đánh giá mô hình**:
   - Metrics: Precision, Recall, F1-score, Confusion Matrix
   - Trên tập test để kiểm tra generalization

6. **Lưu mô hình**:
   - Lưu dưới định dạng pickle (`vehicle_svm_v3.pkl`)
   - Chứa cả scaler và classifier, sẵn sàng sử dụng

---

## Nhiệm Vụ của Mô Hình AI

### **Primary Task: Vehicle Detection**

Mô hình thực hiện **phát hiện xe trong ảnh/video**:

1. **Multi-scale Scanning** (Quét đa tỷ lệ):
   - Sử dụng pyramid của ảnh để phát hiện xe ở các kích thước khác nhau
   - Trượt cửa sổ 64×64 trên mỗi mức pyramid

2. **HOG Feature Extraction** (Trích HOG):
   - Trích xuất HOG cho mỗi patch được quét
   - Chuẩn hóa với scaler đã lưu

3. **SVM Classification** (Phân loại):
   - Dự đoán xác suất patch có chứa xe hay không
   - Nếu confidence > threshold → Ghi nhận detection

4. **Post-Processing** (Xử lý sau):
   - **Heatmap tích lũy**: Gom các detection lặp lại từ nhiều frame liên tiếp
   - **Non-Maximum Suppression (NMS)**: Loại bỏ các box trùng lặp
   - **Filtering**: Loại bỏ detections có confidence thấp

**Output**: Danh sách các bounding box `(x1, y1, x2, y2)` của xe được phát hiện

---

## Ứng Dụng Web - Các Chức Năng

### **Kiến Trúc Ứng Dụng**

Ứng dụng web được xây dựng trên **Flask** với kiến trúc multi-threaded:

```
┌─────────────────────────────────────────┐
│         Web Interface (HTML/JS)         │
│         - Hiển thị video stream         │
│         - Điều khiển camera             │
│         - Hiển thị thống kê              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Flask Web Server (app.py)       │
│  - Route: / (trang chủ)                 │
│  - Route: /video (stream video)         │
│  - Route: /cameras (danh sách camera)   │
│  - Route: /start_camera (chọn camera)   │
└──────────────┬──────────────────────────┘
               │
   ┌───────────┴───────────┐
   │                       │
┌──▼─────────────┐   ┌─────▼──────────────┐
│ Camera Thread  │   │  Detect Thread     │
│ - Capture      │   │ - Vehicle Detection│
│ - Raw frames   │   │ - Post-processing  │
└────────────────┘   └────────────────────┘
   │                       │
   └───────────┬───────────┘
               │
         ┌─────▼────────────┐
         │ Shared State     │
         │ - Raw frame      │
         │ - Detected boxes │
         │ - Car count      │
         └──────────────────┘
```

### **Chức Năng Chi Tiết**

#### 1️. **Real-time Video Streaming** (`/video`)

- Nhận khung hình từ camera
- Vẽ bounding box của các xe được phát hiện
- Hiển thị line đếm (counting line)
- Hiển thị số lượng xe đã đếm
- FPS hiện tại
- Gửi luồng MJPEG tới client qua `multipart/x-mixed-replace`
- **FPS Target**: 20 FPS

#### 2️. **Vehicle Counting** (Đếm xe)

- Người dùng có thể thiết lập một đường tưởng tượng trên video
- Khi tâm của xe **vượt qua** đường này → Tăng bộ đếm lên 1
- Theo dõi số lượng xe đi qua trong khoảng thời gian

#### 3️. **Camera Selection** (`/cameras`, `/start_camera`)

- **Quét tất cả camera** có sẵn trên hệ thống
- **Chọn camera** để cập nhật trong UI
- **Bắt đầu capture** từ camera được chọn
- Hỗ trợ nhiều camera USB, chứ không chỉ webcam mặc định

#### 4️. **Visualization** (Trực quan hóa)

- **Bounding box**: Khoanh tròn vùng xe, mỗi xe màu khác nhau
- **Tracking circle**: Vẽ điểm tâm của mỗi box (để đếm xe)
- **Counting line**: Đường ngang màu đỏ để theo dõi xe qua
- **Statistics overlay**: Hiển thị số lượng xe hiện tại ở góc trên trái

#### 5️. **Multi-threading Architecture**

- **Camera Thread**: Liên tục capture frame từ camera (không chờ detection)
- **Detection Thread**: Chạy độc lập, thực hiện detection trên raw frame
- **Web Server Thread**: Phục vụ HTTP requests, stream video
- **Locking Mechanism**: Đảm bảo thread-safety khi chia sẻ dữ liệu

---

## Cấu Trúc Project

```
car_detection/
├── car_detection.ipynb           # Notebook training model
├── README.md                     # README gốc
├── requirements.txt              # Dependencies
├── app/
│   ├── app.py                    # Entry point Flask
│   ├── camera.py                 # Camera capture & detection threads
│   ├── config.py                 # Configuration
│   ├── detector.py               # Vehicle detection logic
│   ├── routes.py                 # Flask routes
│   ├── state.py                  # Shared state (thread-safe)
│   ├── templates/
│   │   └── index.html            # Web UI
│   └── static/
│       ├── css/
│       │   └── style.css         # Styling
│       └── js/
│           └── main.js           # Frontend JS
└── models/
    └── vehicle_svm_v3.pkl        # Trained SVM model
```

---

## Cách Sử Dụng

### **1. Chuẩn Bị**

```bash
pip install -r requirements.txt
```

### **2. Training Model** (tuỳ chọn)

- Mở file `car_detection.ipynb` trong Jupyter
- Chạy các cell để training từ dataset
- Mô hình sẽ được lưu ở `models/vehicle_svm_v3.pkl`

### **3. Chạy Web App**

```bash
python app/app.py
```

- Server sẽ khởi động tại `http://localhost:5000`
- Truy cập URL trên trong trình duyệt

### **4. Sử Dụng**

- Chọn camera từ dropdown
- Click "Start Camera" để bắt đầu
- Xem video stream với detection
- Theo dõi số lượng xe đi qua đường counting

---

## Hiệu Suất & Tối Ưu Hóa

| Kỹ Thuật                 | Tác Dụng                                 |
| ------------------------ | ---------------------------------------- |
| **OpenCV HOG**           | Nhanh hơn scikit-image ~5x               |
| **HOG Sub-sampling**     | Tính HOG 1 lần cho ROI, tái sử dụng      |
| **Multiprocessing Pool** | Chia pyramid levels cho nhiều CPU cores  |
| **Heatmap tích lũy**     | Giảm False Positive, gom detections      |
| **Detect thread riêng**  | Webcam stream 30 FPS không chờ detection |
| **JPEG encode async**    | Không block main loop                    |

---

## Dependencies

- `flask`: Web framework
- `opencv-python`: Computer vision (HOG, video capture)
- `numpy`: Numerical computing
- `scipy`: Scientific computing (heatmap labeling)
- `scikit-learn`: Machine learning (SVM, preprocessing)
- `joblib`: Model serialization

---

## Tóm Tắt

| Khía Cạnh       | Chi Tiết                                      |
| --------------- | --------------------------------------------- |
| **Mô hình AI**  | HOG + SVM                                     |
| **Công dụng**   | Phát hiện xe trong video thời thực            |
| **Đầu vào**     | Video từ camera                               |
| **Đầu ra**      | Danh sách xe, bounding box, số lượng          |
| **Web app**     | Hiển thị video, chọn camera, đếm xe, thống kê |
| **Performance** | 20 FPS video stream, multi-threaded           |
