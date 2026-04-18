import cv2
import random
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List



#  HẰNG SỐ TOÀN CỤC


IMG_WIDTH  = 128
IMG_HEIGHT = 64
IMG_SIZE   = (IMG_WIDTH, IMG_HEIGHT)   # (w, h) cho cv2.resize

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)



#  CẤU HÌNH AUGMENTATION


@dataclass
class AugConfig:
    """
    Tham số điều chỉnh mức độ augmentation.

    Mặc định được chỉnh cho bài toán phát hiện xe đa góc.
    Chỉnh aug_level = 'light' / 'medium' / 'heavy' để scale nhanh.
    """
    # Rotation — tăng từ ±10 lên ±30 để cover góc camera nghiêng
    rotate_max_deg: float = 30.0

    # Brightness scale (×alpha)
    brightness_range: Tuple[float, float] = (0.5, 1.5)

    # Gamma correction (giả lập overexposed / underexposed)
    gamma_range: Tuple[float, float] = (0.5, 2.0)

    # Perspective — mức độ biến dạng (0=không, 0.3=rất nhiều)
    perspective_strength: float = 0.15

    # Shear — độ nghiêng ngang (giả lập góc nhìn chéo)
    shear_range: float = 0.15

    # Gaussian noise — std độ nhiễu pixel (0–255)
    noise_std: float = 12.0

    # Gaussian blur — kernel size (lẻ), 0 = tắt
    blur_ksize: int = 3

    # Random crop — % chiều bị cắt tối đa mỗi phía
    crop_pct: float = 0.12

    # Horizontal stretch — co/dãn ngang để giả lập xe nhìn ngang / trước
    stretch_range: Tuple[float, float] = (0.75, 1.3)

    @classmethod
    def light(cls) -> "AugConfig":
        """Augmentation nhẹ — dùng khi dataset lớn (>5000 ảnh/lớp)."""
        return cls(rotate_max_deg=15, perspective_strength=0.08,
                   noise_std=6, crop_pct=0.08)

    @classmethod
    def medium(cls) -> "AugConfig":
        """Augmentation mặc định."""
        return cls()

    @classmethod
    def heavy(cls) -> "AugConfig":
        """Augmentation mạnh — dùng khi dataset nhỏ (<1000 ảnh/lớp)."""
        return cls(rotate_max_deg=40, perspective_strength=0.22,
                   noise_std=18, shear_range=0.22, crop_pct=0.18,
                   stretch_range=(0.65, 1.45))


# Mặc định toàn cục (có thể override khi gọi augment_image_v2)
DEFAULT_CFG = AugConfig.medium()



#  TIỀN XỬ LÝ CƠ BẢN


def load_image(path: str) -> Optional[np.ndarray]:
    """Đọc ảnh BGR từ đĩa. Trả None nếu lỗi."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Không đọc được: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """BGR → grayscale. Nếu đã là grayscale thì trả nguyên."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray,
                clip: float = CLAHE_CLIP,
                grid: Tuple[int, int] = CLAHE_GRID) -> np.ndarray:
    """
    CLAHE — cân bằng histogram cục bộ.

    Chia ảnh thành lưới tile, cân bằng độc lập từng tile rồi nội suy bilinear.
    Làm nổi cạnh xe kể cả khi ánh sáng không đồng đều.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)


def resize_image(img: np.ndarray,
                 size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Resize về kích thước chuẩn bằng INTER_AREA (tốt khi thu nhỏ)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa pixel về [0.0, 1.0] (float32).

    Bắt buộc trước khi truyền vào skimage.feature.hog() — tránh overflow
    khi tính gradient trên uint8.
    Nếu đã là float32 trong [0,1] thì trả nguyên bản (tránh chia lại).
    """
    if img.dtype == np.float32 and img.max() <= 1.0:
        return img
    return img.astype(np.float32) / 255.0


def preprocess(img: np.ndarray,
               size: Tuple[int, int] = IMG_SIZE,
               use_clahe: bool = True) -> np.ndarray:
    """
    Pipeline tiền xử lý đầy đủ:
        BGR → resize → grayscale → CLAHE

    Trả về ảnh grayscale uint8 128×64.
    """
    resized = resize_image(img, size)
    gray    = to_grayscale(resized)
    if use_clahe:
        gray = apply_clahe(gray)
    return gray


def preprocess_from_path(path: str, **kwargs) -> Optional[np.ndarray]:
    """Đọc ảnh từ đường dẫn rồi tiền xử lý. Trả None nếu không đọc được."""
    img = load_image(path)
    if img is None:
        return None
    return preprocess(img, **kwargs)



#  CÁC HÀM AUGMENTATION ĐƠN LẺ


def aug_flip(img: np.ndarray) -> np.ndarray:
    """Lật ngang — giả lập xe đi ngược chiều."""
    return cv2.flip(img, 1)


def aug_brightness(img: np.ndarray,
                   lo: float = 0.5, hi: float = 1.5) -> np.ndarray:
    """
    Nhân pixel với alpha ngẫu nhiên.

    Range rộng hơn cũ (0.5–1.5 thay vì 0.7–1.3) để cover
    cảnh ngày nắng và đêm tối hơn.
    """
    alpha = random.uniform(lo, hi)
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def aug_gamma(img: np.ndarray,
              lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """
    Hiệu chỉnh gamma (phi tuyến) — khác với brightness (tuyến tính).

    gamma < 1 → ảnh sáng hơn (overexposed).
    gamma > 1 → ảnh tối hơn (underexposed / ban đêm).
    """
    gamma = random.uniform(lo, hi)
    inv   = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255.0
    lut   = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def aug_rotate(img: np.ndarray, max_deg: float = 30.0) -> np.ndarray:
    """
    Xoay ảnh ±max_deg độ.

    Tăng từ ±10° (cũ) lên ±30° để giả lập camera đặt cao/thấp,
    góc nhìn hơi chếch — giúp nhận ra xe nhìn từ trên xuống.
    """
    h, w  = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def aug_perspective(img: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Biến đổi phối cảnh (perspective transform).

    Dịch chuyển 4 góc ảnh ngẫu nhiên trong phạm vi strength×(w hoặc h).
    Giả lập xe nhìn từ góc trên-bên (bird-eye partial), góc thấp (frog-eye).

    Đây là augmentation quan trọng nhất cho vấn đề đa góc nhìn.
    """
    h, w = img.shape[:2]
    dx   = int(w * strength)
    dy   = int(h * strength)

    # Góc gốc
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Góc đích — dịch chuyển ngẫu nhiên từng góc
    dst = np.float32([
        [random.randint(0, dx),       random.randint(0, dy)],
        [random.randint(w-dx, w),     random.randint(0, dy)],
        [random.randint(w-dx, w),     random.randint(h-dy, h)],
        [random.randint(0, dx),       random.randint(h-dy, h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def aug_shear(img: np.ndarray, shear_range: float = 0.15) -> np.ndarray:
    """
    Shear ngang — nghiêng ảnh sang trái/phải.

    Giả lập xe nhìn từ góc chếch (3/4 view → side view transition).
    """
    h, w  = img.shape[:2]
    shear = random.uniform(-shear_range, shear_range)
    M     = np.float32([[1, shear, 0],
                        [0, 1,     0]])
    # Tính chiều rộng mới sau shear để không bị cắt
    new_w = int(w + abs(shear) * h)
    warped = cv2.warpAffine(img, M, (new_w, h),
                             borderMode=cv2.BORDER_REFLECT)
    # Crop / resize về kích thước gốc
    return cv2.resize(warped, (w, h), interpolation=cv2.INTER_AREA)


def aug_noise(img: np.ndarray, std: float = 12.0) -> np.ndarray:
    """
    Thêm Gaussian noise — giả lập camera chất lượng thấp, điều kiện đêm.
    """
    noise  = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy  = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def aug_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Gaussian blur — giả lập motion blur / camera out-of-focus.

    ksize phải lẻ. Nếu ksize <= 0 thì trả ảnh gốc.
    """
    if ksize <= 0:
        return img
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_crop_resize(img: np.ndarray, crop_pct: float = 0.12) -> np.ndarray:
    """
    Crop ngẫu nhiên rồi resize về kích thước gốc.

    Giả lập detector nhìn thấy xe bị cắt bớt (partial occlusion)
    hoặc xe ở khoảng cách khác trong pyramid.
    """
    h, w   = img.shape[:2]
    max_dx = int(w * crop_pct)
    max_dy = int(h * crop_pct)

    x1 = random.randint(0, max_dx)
    y1 = random.randint(0, max_dy)
    x2 = random.randint(w - max_dx, w)
    y2 = random.randint(h - max_dy, h)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def aug_stretch(img: np.ndarray,
                lo: float = 0.75, hi: float = 1.3) -> np.ndarray:
    """
    Co / dãn ngang rồi crop về kích thước gốc.

    ratio > 1 → dãn rộng → giả lập xe nhìn ngang (side view, dài hơn).
    ratio < 1 → bóp lại   → giả lập xe nhìn gần mặt trước (compact).

    Đây là trick đơn giản nhưng hiệu quả cho vấn đề aspect-ratio.
    """
    h, w  = img.shape[:2]
    ratio = random.uniform(lo, hi)
    new_w = int(w * ratio)
    if new_w < 4:
        return img
    resized = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    if new_w >= w:
        # Dãn rộng → center crop về w
        x_start = (new_w - w) // 2
        return resized[:, x_start:x_start + w]
    else:
        # Bóp lại → pad hai bên bằng reflect
        pad = w - new_w
        left  = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(resized, 0, 0, left, right,
                                  cv2.BORDER_REFLECT)



#  HÀM AUGMENTATION TỔNG HỢP


def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Backward-compatible: trả về đúng 3 biến thể như phiên bản cũ.

    1. Flip ngang
    2. Brightness ±30%  (range cũ)
    3. Rotate ±10°      (range cũ)

    Dùng augment_image_v2() để lấy đầy đủ 9 biến thể.
    """
    return [
        aug_flip(img),
        aug_brightness(img, lo=0.7, hi=1.3),
        aug_rotate(img, max_deg=10.0),
    ]


def augment_image_v2(img: np.ndarray,
                     cfg: AugConfig = DEFAULT_CFG) -> List[np.ndarray]:
    """
    Augmentation nâng cấp — 9 biến thể giải quyết vấn đề đa góc nhìn.

    Thứ tự:
    1.  Flip ngang           — xe đi ngược chiều
    2.  Brightness (±50%)   — ánh sáng mạnh / yếu
    3.  Gamma correction    — over/underexposed
    4.  Rotate ±30°         — camera nghiêng
    5.  Perspective         — góc nhìn từ trên / từ thấp
    6.  Shear ngang         — xe nhìn chếch
    7.  Gaussian noise      — camera chất lượng thấp
    8.  Gaussian blur       — motion blur / mất nét
    9.  Crop + resize       — partial occlusion / scale khác
    10. Stretch ngang       — side view (xe dài) / front view (xe compact)

    Parameters
    ----------
    img : np.ndarray
        Ảnh grayscale uint8 đã qua preprocess() — kích thước IMG_SIZE.
    cfg : AugConfig
        Tham số điều chỉnh. Dùng AugConfig.light/medium/heavy() để chọn nhanh.

    Returns
    -------
    List[np.ndarray]
        9 ảnh augmented (không bao gồm ảnh gốc).
    """
    return [
        aug_flip(img),
        aug_brightness(img, *cfg.brightness_range),
        aug_gamma(img, *cfg.gamma_range),
        aug_rotate(img, cfg.rotate_max_deg),
        aug_perspective(img, cfg.perspective_strength),
        aug_shear(img, cfg.shear_range),
        aug_noise(img, cfg.noise_std),
        aug_blur(img, cfg.blur_ksize),
        aug_crop_resize(img, cfg.crop_pct),
        aug_stretch(img, *cfg.stretch_range),
    ]


def augment_combined(img: np.ndarray,
                     cfg: AugConfig = DEFAULT_CFG,
                     n_extra: int = 3) -> List[np.ndarray]:
    """
    Tạo thêm biến thể bằng cách KẾT HỢP ngẫu nhiên nhiều phép biến đổi.

    Ví dụ: flip + brightness + noise cùng lúc → gần với thực tế hơn
    vì ảnh thật thường có nhiều yếu tố cùng xuất hiện.

    Parameters
    ----------
    img     : np.ndarray  — ảnh gốc đã preprocess
    cfg     : AugConfig
    n_extra : int         — số ảnh kết hợp cần tạo thêm

    Returns
    -------
    List[np.ndarray]  — n_extra ảnh (không bao gồm gốc)
    """
    # Danh sách các phép đơn lẻ (không dùng flip để tránh double-flip)
    ops = [
        lambda x: aug_brightness(x, *cfg.brightness_range),
        lambda x: aug_gamma(x, *cfg.gamma_range),
        lambda x: aug_rotate(x, cfg.rotate_max_deg),
        lambda x: aug_perspective(x, cfg.perspective_strength),
        lambda x: aug_shear(x, cfg.shear_range),
        lambda x: aug_noise(x, cfg.noise_std),
        lambda x: aug_blur(x, cfg.blur_ksize),
        lambda x: aug_crop_resize(x, cfg.crop_pct),
        lambda x: aug_stretch(x, *cfg.stretch_range),
    ]

    results = []
    for _ in range(n_extra):
        out = img.copy()
        # Áp dụng 2–3 phép ngẫu nhiên liên tiếp
        for op in random.sample(ops, k=random.randint(2, 3)):
            out = op(out)
        results.append(out)
    return results



#  LOAD DATASET


def load_dataset(pos_dir: str,
                 neg_dir: str,
                 use_clahe: bool = True,
                 max_per_class: Optional[int] = None,
                 augment: bool = False,
                 augment_v2: bool = False,
                 aug_cfg: AugConfig = DEFAULT_CFG,
                 n_combined: int = 0,
                 augment_pos_only: bool = False):
    """
    Tải toàn bộ dataset từ 2 thư mục positive / negative.

    Parameters
    ----------
    pos_dir          : str        — thư mục ảnh có xe
    neg_dir          : str        — thư mục ảnh không có xe
    use_clahe        : bool       — áp dụng CLAHE
    max_per_class    : int | None — giới hạn số ảnh mỗi lớp
    augment          : bool       — dùng augment_image() cũ (3 biến thể)
    augment_v2       : bool       — dùng augment_image_v2() mới (9 biến thể) ← KHUYẾN NGHỊ
    aug_cfg          : AugConfig  — cấu hình augmentation v2
    n_combined       : int        — số ảnh kết hợp thêm (augment_combined)
    augment_pos_only : bool       — chỉ augment positive (tiết kiệm bộ nhớ)

    Ưu tiên: augment_v2 > augment > không augment.

    Returns
    -------
    images : list[np.ndarray]
    labels : list[int]  — 1 = có xe, 0 = không có xe
    """
    images, labels = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def _load_from_dir(directory: str, label: int):
        paths = sorted(p for p in Path(directory).rglob("*")
                       if p.suffix.lower() in extensions)
        if max_per_class:
            paths = paths[:max_per_class]

        do_aug = augment_v2 or augment
        # Nếu augment_pos_only=True thì chỉ augment positive (label=1)
        do_aug = do_aug and (not augment_pos_only or label == 1)

        n_aug_per_img = 0
        for p in paths:
            processed = preprocess_from_path(str(p), use_clahe=use_clahe)
            if processed is None:
                continue

            images.append(processed)
            labels.append(label)

            if do_aug:
                if augment_v2:
                    variants = augment_image_v2(processed, aug_cfg)
                else:
                    variants = augment_image(processed)

                if n_combined > 0:
                    variants += augment_combined(processed, aug_cfg, n_combined)

                n_aug_per_img = len(variants)
                for v in variants:
                    images.append(v)
                    labels.append(label)

        tag        = "[POS]" if label == 1 else "[NEG]"
        total_aug  = len(paths) * (1 + n_aug_per_img) if do_aug else len(paths)
        aug_note   = (f" → ×{1 + n_aug_per_img} = {total_aug} ảnh"
                      if do_aug else "")
        print(f"  {tag} {len(paths)} ảnh gốc{aug_note}  [{directory}]")

    print("─" * 55)
    print("Đang tải dataset...")

    if augment_v2:
        print(f"  Chế độ: augment_v2 (9 biến thể/ảnh)"
              + (f" + {n_combined} combined" if n_combined else "")
              + (" | pos only" if augment_pos_only else ""))
    elif augment:
        print("  Chế độ: augment v1 (3 biến thể/ảnh)")
    else:
        print("  Chế độ: không augment")

    _load_from_dir(pos_dir, label=1)
    _load_from_dir(neg_dir, label=0)

    n_pos = labels.count(1)
    n_neg = labels.count(0)
    print(f"  Tổng: {len(images)} ảnh  (pos={n_pos}, neg={n_neg})")
    if n_pos and n_neg:
        ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
        if ratio > 2.0:
            print(f"  [WARN] Mất cân bằng lớp: {ratio:.1f}x — "
                  "cân nhắc class_weight='balanced' trong SVM")
    print("─" * 55)

    return images, labels
