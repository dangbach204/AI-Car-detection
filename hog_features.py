"""
hog_features.py
===============
Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ ảnh.

Nguyên lý HOG
─────────────
1. Tính gradient (∂I/∂x, ∂I/∂y) tại từng pixel → độ lớn & hướng.
2. Chia ảnh thành các "cell" nhỏ (e.g. 8×8 pixel).
3. Trong mỗi cell, tạo histogram 9 bin theo hướng gradient (0°–180°).
4. Nhóm các cell thành "block" (2×2 cells) → chuẩn hóa L2 để bất biến ánh sáng.
5. Nối tất cả histogram đã chuẩn hóa thành một vector đặc trưng dài.

Với ảnh 64×128, vector HOG có độ dài = 3780 chiều (scikit-image default).
"""

import numpy as np
from skimage.feature import hog
from skimage import exposure
from typing import Tuple, Optional
import cv2


# ── Tham số HOG (đề xuất cho phát hiện xe) ─────────────────────────────────
HOG_PARAMS = dict(
    orientations=12,          # tăng từ 9 → 12
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),   # tăng từ 2×2 → 3×3
    block_norm="L2-Hys",
    visualize=False,
    feature_vector=True,
    channel_axis=None
)


def extract_hog(image: np.ndarray,
                visualize: bool = False,
                params: Optional[dict] = None) -> np.ndarray:
    """
    Trích xuất đặc trưng HOG từ một ảnh grayscale.

    Parameters
    ----------
    image     : np.ndarray (H×W, dtype uint8 hoặc float)
        Ảnh grayscale đã tiền xử lý (64×128 px).
    visualize : bool
        Nếu True, trả về cả ảnh HOG để hiển thị.
    params    : dict, optional
        Ghi đè tham số HOG mặc định.

    Returns
    -------
    features  : np.ndarray (1D)
        Vector đặc trưng HOG.
    hog_image : np.ndarray (chỉ khi visualize=True)
        Ảnh trực quan hóa HOG.
    """
    p = {**HOG_PARAMS, **(params or {})}

    # Đảm bảo float32 để tính nhanh hơn
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    if visualize:
        p["visualize"] = True
        features, hog_image = hog(image, **p)
        # Rescale để hiển thị rõ hơn
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return features, hog_image
    else:
        p["visualize"] = False
        return hog(image, **p)


def extract_hog_batch(images: list,
                      params: Optional[dict] = None,
                      verbose: bool = True) -> np.ndarray:
    """
    Trích xuất HOG cho một danh sách ảnh.

    Parameters
    ----------
    images  : list[np.ndarray]
        Danh sách ảnh grayscale đã tiền xử lý.
    params  : dict, optional
        Ghi đè tham số HOG.
    verbose : bool
        In tiến trình.

    Returns
    -------
    np.ndarray (N × D)
        Ma trận đặc trưng, N = số ảnh, D = độ dài vector HOG.
    """
    if verbose:
        print(f"Trích xuất HOG cho {len(images)} ảnh...")

    features_list = []
    for i, img in enumerate(images):
        feat = extract_hog(img, visualize=False, params=params)
        features_list.append(feat)

        if verbose and (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(images)}] "
                  f"Vector size = {feat.shape[0]}")

    X = np.array(features_list, dtype=np.float32)

    if verbose:
        print(f"  Hoàn thành! Ma trận đặc trưng: {X.shape} "
              f"({X.nbytes / 1e6:.1f} MB)")
    return X


def hog_feature_size(img_size=(64, 128), params: Optional[dict] = None) -> int:
    """
    Tính trước kích thước vector HOG mà không cần ảnh thật.

    Parameters
    ----------
    img_size : (width, height)
    params   : dict — tham số HOG tùy chỉnh

    Returns
    -------
    int — số chiều của vector HOG
    """
    dummy = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
    return extract_hog(dummy, params=params).shape[0]


# ── Sliding window HOG (dùng trong detector) ────────────────────────────────

def sliding_window_hog(image: np.ndarray,
                       win_size: Tuple[int, int] = (64, 128),
                       step_size: int = 8,
                       params: Optional[dict] = None):
    """
    Generator: trượt cửa sổ HOG qua toàn bộ ảnh.

    Yields
    ------
    (x, y, features) : (int, int, np.ndarray)
        Tọa độ góc trên-trái và vector HOG của vùng đó.
    """
    win_w, win_h = win_size
    h, w = image.shape[:2]

    for y in range(0, h - win_h + 1, step_size):
        for x in range(0, w - win_w + 1, step_size):
            window = image[y:y + win_h, x:x + win_w]
            # Grayscale nếu cần
            if len(window.shape) == 3:
                window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            feat = extract_hog(window, params=params)
            yield x, y, feat