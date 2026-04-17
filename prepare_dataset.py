"""
prepare_dataset.py
==================
Chuyển đổi dataset YOLO format (Stanford Cars từ Kaggle)
→ thư mục positive/ và negative/ để train HOG + SVM

Cấu trúc dataset đầu vào (YOLO format):
    data_raw/
    ├── train/
    │   ├── images/   ← file ảnh .jpg / .png
    │   └── labels/   ← file nhãn .txt (YOLO format)
    ├── valid/        (hoặc "value")
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Format mỗi dòng trong file .txt (YOLO):
    class_id  x_center  y_center  width  height
    (tất cả đã chuẩn hóa 0–1 so với kích thước ảnh)

Kết quả đầu ra:
    data/
    ├── positive/   ← crop vùng CÓ XE (64×128 px)
    └── negative/   ← crop vùng KHÔNG CÓ XE (64×128 px)

Cách dùng:
    python prepare_dataset.py --raw car_dataset-master --out data --splits train valid test
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm


# ── Kích thước ảnh chuẩn cho HOG ────────────────────────────────────────────
TARGET_W = 64
TARGET_H = 128
TARGET_SIZE = (TARGET_W, TARGET_H)

# ── Padding khi crop ảnh xe (để có thêm context) ─────────────────────────────
CROP_PADDING = 0.05   # thêm 5% mỗi phía


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def yolo_to_pixel(x_c, y_c, w, h, img_w, img_h):
    """
    Chuyển tọa độ YOLO (chuẩn hóa) → pixel tuyệt đối.

    Returns: (x1, y1, x2, y2)
    """
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2


def crop_with_padding(img, x1, y1, x2, y2, pad=CROP_PADDING):
    """
    Crop bbox có thêm padding, clip vào biên ảnh.
    """
    ih, iw = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad)
    py = int(bh * pad)

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px)
    cy2 = min(ih, y2 + py)

    return img[cy1:cy2, cx1:cx2]


def random_negative_crop(img, bboxes, size=(TARGET_W, TARGET_H),
                          max_tries=30, iou_thresh=0.1):
    """
    Lấy vùng crop ngẫu nhiên KHÔNG trùng (IoU < thresh) với bất kỳ bbox nào.

    Parameters
    ----------
    img      : np.ndarray
    bboxes   : list of (x1, y1, x2, y2) — pixel coords
    size     : (w, h) — kích thước crop muốn lấy (trước khi resize)
    max_tries: số lần thử tối đa

    Returns
    -------
    np.ndarray hoặc None nếu không tìm được
    """
    ih, iw = img.shape[:2]
    cw, ch = size

    # Tỷ lệ crop theo kích thước ảnh
    crop_w = max(cw, iw // 4)
    crop_h = max(ch, ih // 4)

    if iw <= crop_w or ih <= crop_h:
        return None

    for _ in range(max_tries):
        rx = random.randint(0, iw - crop_w)
        ry = random.randint(0, ih - crop_h)
        rx2 = rx + crop_w
        ry2 = ry + crop_h

        # Kiểm tra IoU với từng bbox
        overlap = False
        for bx1, by1, bx2, by2 in bboxes:
            ix1 = max(rx,  bx1)
            iy1 = max(ry,  by1)
            ix2 = min(rx2, bx2)
            iy2 = min(ry2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_crop = crop_w * crop_h
            if inter / area_crop > iou_thresh:
                overlap = True
                break

        if not overlap:
            return img[ry:ry2, rx:rx2]

    return None


def save_image(crop, out_path):
    """Resize về TARGET_SIZE rồi lưu."""
    if crop is None or crop.size == 0:
        return False
    resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), resized)
    return True


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def process_split(split_dir: Path,
                  pos_dir: Path,
                  neg_dir: Path,
                  max_per_class: int,
                  pos_counter: list,
                  neg_counter: list):
    """
    Xử lý một split (train / valid / test).
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # Một số dataset dùng "value" thay vì "valid"
    if not images_dir.exists():
        print(f"  [SKIP] Không tìm thấy: {images_dir}")
        return

    img_paths = sorted(list(images_dir.glob("*.jpg")) +
                       list(images_dir.glob("*.jpeg")) +
                       list(images_dir.glob("*.png")))

    print(f"\n  [{split_dir.name}] {len(img_paths)} ảnh")

    for img_path in tqdm(img_paths, desc=f"  {split_dir.name}", ncols=80):

        # Kiểm tra giới hạn
        if max_per_class and (pos_counter[0] >= max_per_class and
                               neg_counter[0] >= max_per_class):
            break

        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        # Đọc file nhãn tương ứng
        label_path = labels_dir / (img_path.stem + ".txt")
        bboxes_pixel = []

        if label_path.exists():
            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    x1, y1, x2, y2 = yolo_to_pixel(x_c, y_c, w, h, iw, ih)
                    bboxes_pixel.append((x1, y1, x2, y2))

                    # ── Positive: crop vùng có xe ──────────────────────────
                    if not max_per_class or pos_counter[0] < max_per_class:
                        crop = crop_with_padding(img, x1, y1, x2, y2)
                        if crop.size > 0:
                            out = pos_dir / f"{img_path.stem}_{pos_counter[0]:06d}.jpg"
                            if save_image(crop, out):
                                pos_counter[0] += 1

                except (ValueError, IndexError):
                    continue

        # ── Negative: crop vùng ngẫu nhiên không có xe ─────────────────────
        if not max_per_class or neg_counter[0] < max_per_class:
            n_neg = max(3, len(bboxes_pixel) * 3)   # số negative = số positive trong ảnh
            for _ in range(n_neg):
                if max_per_class and neg_counter[0] >= max_per_class:
                    break
                neg_crop = random_negative_crop(img, bboxes_pixel)
                if neg_crop is not None:
                    out = neg_dir / f"{img_path.stem}_neg_{neg_counter[0]:06d}.jpg"
                    if save_image(neg_crop, out):
                        neg_counter[0] += 1


def prepare(raw_dir: str,
            out_dir: str,
            splits: list,
            max_per_class: int = None):
    """
    Hàm chính: chuyển đổi toàn bộ dataset.
    """
    raw_path = Path(raw_dir)
    pos_dir  = Path(out_dir) / "positive"
    neg_dir  = Path(out_dir) / "negative"

    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CHUẨN BỊ DATASET HOG + SVM")
    print("=" * 60)
    print(f"  Nguồn  : {raw_path}")
    print(f"  Đích   : {Path(out_dir)}")
    print(f"  Splits : {splits}")
    print(f"  Max/lớp: {max_per_class or 'không giới hạn'}")

    pos_counter = [0]   # dùng list để truyền tham chiếu
    neg_counter = [0]

    for split in splits:
        # Thử cả "valid" và "value" (tên hay bị nhầm)
        for split_name in [split, split.replace("valid", "value"),
                            split.replace("value", "valid")]:
            split_dir = raw_path / split_name
            if split_dir.exists():
                process_split(split_dir, pos_dir, neg_dir,
                              max_per_class, pos_counter, neg_counter)
                break
        else:
            print(f"\n  [WARN] Không tìm thấy split: {split} (đã thử các tên)")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print(f"  positive/ : {pos_counter[0]:,} ảnh  →  {pos_dir}")
    print(f"  negative/ : {neg_counter[0]:,} ảnh  →  {neg_dir}")
    print("=" * 60)
    print("\nBước tiếp theo:")
    print("  python train.py --pos data/positive --neg data/negative --output models")


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuyển YOLO dataset → positive/negative cho HOG+SVM"
    )
    parser.add_argument(
        "--raw", required=True,
        help="Thư mục gốc chứa dataset (có thư mục train/valid/test bên trong)"
    )
    parser.add_argument(
        "--out", default="data",
        help="Thư mục đầu ra (sẽ tạo data/positive và data/negative)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        help="Các split cần xử lý (mặc định: train valid test)"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Số ảnh tối đa mỗi lớp (không đặt = lấy hết ~8000 pos + 8000 neg)"
    )
    args = parser.parse_args()

    prepare(
        raw_dir=args.raw,
        out_dir=args.out,
        splits=args.splits,
        max_per_class=args.max,
    )