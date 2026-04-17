"""
preprocess.py
=============
Tiền xử lý ảnh cho pipeline HOG + SVM phát hiện xe.

Các bước:
1. Resize về 64×128 pixel (chuẩn HOG người đi bộ, áp dụng cho xe)
2. Chuyển sang grayscale
3. Cân bằng sáng bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. Normalize pixel về [0, 1] (chuẩn bị cho HOG)
5. Tăng cường dữ liệu (Data Augmentation) — tùy chọn
"""

import cv2
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ── Hằng số toàn cục ───────────────────────────────────────────────────────
IMG_WIDTH  = 64    # chiều rộng chuẩn (px)
IMG_HEIGHT = 128   # chiều cao chuẩn (px)
IMG_SIZE   = (IMG_WIDTH, IMG_HEIGHT)  # (w, h) cho cv2.resize

CLAHE_CLIP    = 2.0    # ngưỡng clip của CLAHE
CLAHE_GRID    = (8, 8) # kích thước lưới tile


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Đọc ảnh từ đĩa. Trả về None nếu không đọc được.

    Parameters
    ----------
    path : str
        Đường dẫn tới file ảnh.

    Returns
    -------
    np.ndarray hoặc None
        Ảnh BGR (OpenCV) hoặc None nếu lỗi.
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Không đọc được: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh BGR → grayscale."""
    if len(img.shape) == 2:
        return img  # đã là grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Cân bằng histogram thích nghi CLAHE.

    CLAHE chia ảnh thành các ô nhỏ (tile) và cân bằng độc lập từng ô,
    giúp làm nổi rõ cạnh và kết cấu bề mặt xe kể cả khi ánh sáng không đều.

    Parameters
    ----------
    gray : np.ndarray
        Ảnh grayscale đầu vào (uint8).

    Returns
    -------
    np.ndarray
        Ảnh đã qua CLAHE (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(gray)


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize giá trị pixel về khoảng [0.0, 1.0].

    Chuyển ảnh uint8 (0–255) sang float32 (0.0–1.0) bằng cách chia cho 255.
    Bước này cần thiết trước khi đưa ảnh vào hàm extract_hog() của
    scikit-image, vốn yêu cầu đầu vào kiểu float trong khoảng [0, 1].

    Parameters
    ----------
    img : np.ndarray (dtype uint8 hoặc float)
        Ảnh grayscale đã qua CLAHE.

    Returns
    -------
    np.ndarray (dtype float32, shape không đổi)
        Ảnh với giá trị pixel trong [0.0, 1.0].

    Notes
    -----
    - Nếu img đã là float32 trong [0,1] thì hàm trả về nguyên bản (tránh
      chia lại gây mất độ chính xác).
    - StandardScaler trong train.py là bước normalize ở cấp độ VECTOR HOG
      (zero-mean, unit-variance), hoàn toàn độc lập với bước này.
    """
    if img.dtype == np.float32 and img.max() <= 1.0:
        return img  # đã normalize rồi, bỏ qua
    return img.astype(np.float32) / 255.0


def resize_image(img: np.ndarray,
                 size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Resize ảnh về kích thước chuẩn bằng INTER_AREA (tốt cho thu nhỏ).

    Parameters
    ----------
    img  : np.ndarray
        Ảnh đầu vào (bất kỳ kích thước).
    size : (width, height)
        Kích thước đích.

    Returns
    -------
    np.ndarray
        Ảnh đã resize.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def preprocess(img: np.ndarray,
               size: Tuple[int, int] = IMG_SIZE,
               use_clahe: bool = True,
               do_normalize: bool = True) -> np.ndarray:
    """
    Pipeline tiền xử lý đầy đủ:
        BGR → resize → grayscale → CLAHE → normalize [0,1]

    Parameters
    ----------
    img          : np.ndarray
        Ảnh BGR gốc.
    size         : tuple
        Kích thước đích (width, height).
    use_clahe    : bool
        Có áp dụng CLAHE không (mặc định True).
    do_normalize : bool
        Có normalize pixel về [0,1] không (mặc định True).

    Returns
    -------
    np.ndarray (float32, shape 128×64)
        Ảnh grayscale đã chuẩn hóa, sẵn sàng cho extract_hog().
    """
    # 1. Resize
    resized = resize_image(img, size)

    # 2. Grayscale
    gray = to_grayscale(resized)

    # 3. CLAHE (trên uint8)
    if use_clahe:
        gray = apply_clahe(gray)

    # 4. Normalize pixel → float32 [0, 1]
    if do_normalize:
        gray = normalize(gray)

    return gray


def preprocess_from_path(path: str, **kwargs) -> Optional[np.ndarray]:
    """
    Đọc ảnh từ đường dẫn rồi tiền xử lý.

    Returns None nếu không đọc được file.
    """
    img = load_image(path)
    if img is None:
        return None
    return preprocess(img, **kwargs)


# ── Data Augmentation ───────────────────────────────────────────────────────

def augment_image(img: np.ndarray) -> list:
    """
    Tăng cường dữ liệu: tạo thêm ảnh biến thể từ ảnh gốc.

    Các phép biến đổi:
    1. Lật ngang (Horizontal Flip)      — mô phỏng xe đi từ phải sang trái
    2. Thay đổi độ sáng ngẫu nhiên      — mô phỏng điều kiện ánh sáng khác nhau
    3. Xoay nhẹ ±10 độ                  — mô phỏng camera đặt hơi nghiêng

    Parameters
    ----------
    img : np.ndarray
        Ảnh grayscale float32 [0,1] đã tiền xử lý đầy đủ.

    Returns
    -------
    list[np.ndarray]
        Danh sách 3 ảnh đã augment (không bao gồm ảnh gốc).
    """
    results = []

    # Đảm bảo float32 để tính toán nhất quán
    img_f = img.astype(np.float32)

    # 1. Lật ngang (Horizontal Flip)
    flipped = cv2.flip(img_f, 1)
    results.append(flipped)

    # 2. Thay đổi độ sáng ngẫu nhiên (±30%) — clip trong [0, 1]
    alpha = random.uniform(0.7, 1.3)
    bright = np.clip(img_f * alpha, 0.0, 1.0).astype(np.float32)
    results.append(bright)

    # 3. Xoay nhẹ ±10 độ
    h, w = img_f.shape[:2]
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_f, M, (w, h),
                              borderMode=cv2.BORDER_REFLECT)
    results.append(rotated)

    return results


# ── Load Dataset ────────────────────────────────────────────────────────────

def load_dataset(pos_dir: str,
                 neg_dir: str,
                 use_clahe: bool = True,
                 do_normalize: bool = True,
                 max_per_class: Optional[int] = None,
                 augment: bool = False):
    """
    Tải toàn bộ dataset từ 2 thư mục positive/negative.

    Parameters
    ----------
    pos_dir       : str  — thư mục chứa ảnh có xe
    neg_dir       : str  — thư mục chứa ảnh không có xe
    use_clahe     : bool — áp dụng CLAHE hay không
    do_normalize  : bool — normalize pixel về [0,1] (mặc định True)
    max_per_class : int  — giới hạn số ảnh mỗi lớp (None = lấy hết)
    augment       : bool — có áp dụng data augmentation cho positive không

    Returns
    -------
    images  : list[np.ndarray float32]  — danh sách ảnh đã tiền xử lý
    labels  : list[int]                 — 1 = có xe, 0 = không có xe
    """
    images, labels = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def _load_from_dir(directory: str, label: int):
        paths = [p for p in Path(directory).rglob("*")
                 if p.suffix.lower() in extensions]
        if max_per_class:
            paths = paths[:max_per_class]

        for p in paths:
            processed = preprocess_from_path(str(p),
                                             use_clahe=use_clahe,
                                             do_normalize=do_normalize)
            if processed is not None:
                images.append(processed)
                labels.append(label)

                # Augmentation chỉ áp dụng cho positive (ảnh có xe)
                if augment and label == 1:
                    for aug_img in augment_image(processed):
                        images.append(aug_img)
                        labels.append(label)

        print(f"  {'[POS]' if label else '[NEG]'} "
              f"Đã tải {len(paths)} ảnh từ {directory}"
              + (f" → augment x3 = {len(paths)*4} ảnh" if augment and label == 1 else ""))

    print("─" * 50)
    print("Đang tải dataset...")
    _load_from_dir(pos_dir, label=1)
    _load_from_dir(neg_dir, label=0)
    print(f"  Tổng cộng: {len(images)} ảnh "
          f"({labels.count(1)} positive, {labels.count(0)} negative)")
    print("─" * 50)

    return images, labels
