"""
detector.py
===========
Phát hiện xe trong ảnh/frame bằng:
    1. Global Rejection Gate  — kiểm tra toàn ảnh trước, không xe → bỏ qua
    2. Image Pyramid          — xử lý ở nhiều tỷ lệ để bắt xe to/nhỏ
    3. Sliding Window         — trượt cửa sổ HOG qua toàn bộ ảnh
    4. HOG + SVM              — phân loại từng vùng
    5. NMS                    — loại bỏ bbox trùng lặp
    6. Merge Boxes            — gộp các bbox gần nhau thành 1 bbox bao quanh xe

Đầu ra: danh sách bounding box [(x1, y1, x2, y2, score), ...]
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import apply_clahe, IMG_SIZE
from hog_features import extract_hog


# ── Kiểu dữ liệu ────────────────────────────────────────────────────────────
BBox = Tuple[int, int, int, int, float]   # (x1, y1, x2, y2, score)


# ── Tham số phát hiện mặc định ──────────────────────────────────────────────
DETECT_CFG = dict(
    win_size         = (128, 64),   # (width, height) — cùng kích thước train
    step_size        = 8,           # bước trượt (px) — nhỏ hơn → chính xác hơn, chậm hơn
    scale_factor     = 1.25,        # hệ số thu nhỏ mỗi tầng pyramid
    min_wh           = (64, 32),    # kích thước window tối thiểu (px)
    nms_thresh       = 0.3,         # ngưỡng IoU cho NMS
    conf_thresh      = 0.90,        # ngưỡng xác suất phát hiện
    merge_thresh     = 0.15,        # ngưỡng IoU để gộp box gần nhau

    # ── MỚI: Global Rejection Gate ──────────────────────────────────────────
    # Trước khi chạy sliding window, resize toàn ảnh về win_size và phân loại.
    # Nếu score < global_reject_thresh → KHÔNG chạy sliding window → return []
    #
    # Giúp loại bỏ false positive khi ảnh chứa vật thể không phải xe
    # (hổ, người, cây...) mà tình cờ có sub-region giống HOG xe.
    use_global_gate      = True,    # Bật/tắt tính năng
    global_reject_thresh = 0.40,    # Ngưỡng reject (0.25–0.50 tuỳ dataset)
)



#  IMAGE PYRAMID


def image_pyramid(image: np.ndarray,
                  scale: float = 1.25,
                  min_size: Tuple[int, int] = (64, 128)):
    """
    Generator: tạo chuỗi ảnh thu nhỏ dần (pyramid).

    Yield
    -----
    (level_image, current_scale)
        level_image   : np.ndarray — ảnh ở tầng hiện tại
        current_scale : float      — tỷ lệ so với ảnh gốc
    """
    current = image.copy()
    current_scale = 1.0

    while True:
        yield current, current_scale
        h, w = current.shape[:2]

        if w / scale < min_size[0] or h / scale < min_size[1]:
            break

        new_w = int(w / scale)
        new_h = int(h / scale)
        current = cv2.resize(current, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        current_scale *= scale



#  SLIDING WINDOW


def sliding_window(image: np.ndarray,
                   win_size: Tuple[int, int],
                   step: int):
    """
    Generator: trượt cửa sổ qua ảnh.

    Yield
    -----
    (x, y, window)
        x, y   : int         — góc trên-trái
        window : np.ndarray  — vùng cắt ra (win_h × win_w)
    """
    win_w, win_h = win_size
    h, w = image.shape[:2]

    for y in range(0, h - win_h + 1, step):
        for x in range(0, w - win_w + 1, step):
            yield x, y, image[y:y + win_h, x:x + win_w]



#  NMS — Non-Maximum Suppression


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Tính IoU (Intersection over Union) giữa 2 bbox."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray,
                        scores: np.ndarray,
                        iou_threshold: float = 0.1) -> List[int]:
    """
    Thuật toán NMS (Greedy).

    Parameters
    ----------
    boxes         : np.ndarray (N × 4) — [x1, y1, x2, y2]
    scores        : np.ndarray (N,)    — confidence score
    iou_threshold : float

    Returns
    -------
    List[int] — chỉ số của các bbox được giữ lại
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    kept  = []

    while len(order) > 0:
        i = order[0]
        kept.append(i)

        if len(order) == 1:
            break

        ious = np.array([compute_iou(boxes[i], boxes[j])
                         for j in order[1:]])
        order = order[1:][ious <= iou_threshold]

    return kept



#  MERGE BOXES


def _coverage(b_small, b_large) -> float:
    """Tỷ lệ diện tích b_small bị b_large che phủ."""
    ix1 = max(b_small[0], b_large[0])
    iy1 = max(b_small[1], b_large[1])
    ix2 = min(b_small[2], b_large[2])
    iy2 = min(b_small[3], b_large[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (b_small[2] - b_small[0]) * (b_small[3] - b_small[1]))
    return inter / area


def coverage_nms(boxes: List, scores: List,
                 coverage_thresh: float = 0.45) -> List[BBox]:
    """Coverage-NMS: loại box nhỏ bị box lớn che phủ ≥ coverage_thresh."""
    if len(boxes) == 0:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        bx = boxes[i]

        for j in order:
            if j in suppressed or j == i:
                continue
            if _coverage(boxes[j], bx) >= coverage_thresh:
                suppressed.add(j)

    return [(int(boxes[i][0]), int(boxes[i][1]),
             int(boxes[i][2]), int(boxes[i][3]),
             float(scores[i])) for i in kept]


def merge_boxes(boxes: np.ndarray,
                scores: np.ndarray,
                nms_threshold: float = 0.3,
                merge_threshold: float = 0.05) -> List[BBox]:
    """Gộp các bbox cùng xe thành 1 bbox lớn bao quanh xe (3 bước)."""
    if len(boxes) == 0:
        return []

    # Bước 1: IoU-NMS
    kept_idx = non_max_suppression(boxes, scores, nms_threshold)
    boxes_k  = boxes[kept_idx].tolist()
    scores_k = scores[kept_idx].tolist()
    if not boxes_k:
        return []

    # Bước 2: Coverage-NMS
    after_cov = coverage_nms(boxes_k, scores_k, coverage_thresh=0.45)
    if not after_cov:
        return []

    # Bước 3: Union-merge
    boxes_m  = [[b[0], b[1], b[2], b[3]] for b in after_cov]
    scores_m = [b[4] for b in after_cov]
    merged = []
    used   = set()

    for i in range(len(boxes_m)):
        if i in used:
            continue
        group_b = [boxes_m[i]]
        group_s = [scores_m[i]]
        used.add(i)

        for j in range(i + 1, len(boxes_m)):
            if j in used:
                continue
            cov_ij = _coverage(boxes_m[j], boxes_m[i])
            cov_ji = _coverage(boxes_m[i], boxes_m[j])
            if max(cov_ij, cov_ji) >= merge_threshold:
                group_b.append(boxes_m[j])
                group_s.append(scores_m[j])
                used.add(j)

        arr = np.array(group_b)
        merged.append((
            int(arr[:, 0].min()), int(arr[:, 1].min()),
            int(arr[:, 2].max()), int(arr[:, 3].max()),
            float(max(group_s))
        ))

    return merged


def remove_nested_boxes(boxes, coverage_thresh: float = 0.45):
    """Lần quét cuối: xóa box nhỏ bị box lớn hơn che phủ ≥ coverage_thresh."""
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)
    final = []
    for b in boxes_sorted:
        if not any(_coverage(b, fb) >= coverage_thresh for fb in final):
            final.append(b)
    return final



#  VEHICLE DETECTOR


class VehicleDetector:
    """
    Bộ phát hiện xe dùng HOG + SVM + Sliding Window + Image Pyramid.

    Sử dụng:
        detector = VehicleDetector(clf, scaler, cfg)
        boxes = detector.detect(frame)
    """

    def __init__(self, clf, scaler, cfg: dict = None):
        self.clf    = clf
        self.scaler = scaler
        self.cfg    = {**DETECT_CFG, **(cfg or {})}
        self._has_proba = hasattr(clf, "predict_proba")

    # ══════════════════════════════════════════════════════════════════════
    #  MỚI: GLOBAL REJECTION GATE
    # ══════════════════════════════════════════════════════════════════════
    def _global_gate(self, gray: np.ndarray) -> float:
        """
        Resize toàn ảnh về win_size → HOG → SVM → trả về score [0, 1].

        Cách hoạt động:
            1. Resize toàn bộ ảnh (bất kể kích thước) về win_size (128×64)
            2. Trích HOG → vector đặc trưng
            3. Chạy SVM → lấy xác suất lớp "xe"

        Ý nghĩa:
            - Score cao (>= 0.40): ảnh "trông giống xe" → tiếp tục sliding window
            - Score thấp (< 0.40): ảnh không phải xe → return [] ngay

        Lưu ý:
            - Nếu xe nhỏ trong ảnh góc rộng, score có thể thấp giả
            - Trường hợp đó: giảm global_reject_thresh xuống 0.25–0.30
        """
        win_w, win_h = self.cfg["win_size"]
        resized = cv2.resize(gray, (win_w, win_h))
        feat    = extract_hog(resized)
        feat    = self.scaler.transform(feat.reshape(1, -1))

        if self._has_proba:
            return float(self.clf.predict_proba(feat)[0][1])
        else:
            d = self.clf.decision_function(feat)[0]
            return float(1 / (1 + np.exp(-d)))  # sigmoid

    # ── Phát hiện trong một ảnh ────────────────────────────────────────────
    def detect(self,
               frame: np.ndarray,
               return_all: bool = False) -> List[BBox]:
        """
        Phát hiện xe trong một frame.

        Parameters
        ----------
        frame      : np.ndarray (BGR)
        return_all : bool — nếu True, trả về cả bbox trước NMS (debug)

        Returns
        -------
        List[(x1, y1, x2, y2, score)] — bbox sau NMS + Merge
        """
        cfg = self.cfg

        # Chuyển sang grayscale + CLAHE
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        gray = apply_clahe(gray)

        # ══════════════════════════════════════════════════════════════════
        #  BƯỚC 0: GLOBAL REJECTION GATE  ← MỚI
        #
        #  Thuật toán:
        #    global_score = SVM(HOG(resize(toàn_ảnh, win_size)))
        #    if global_score < global_reject_thresh:
        #        return []   ← không chạy sliding window
        #
        #  Tại sao fix được lỗi ảnh hổ:
        #    - Classifier trên TOÀN ẢNH hổ → score ~15.75%
        #    - 15.75% < 40% (threshold) → gate từ chối
        #    - Sliding window KHÔNG chạy → không có false positive 97.39%
        #
        #  Tại sao an toàn với ảnh xe thật:
        #    - Ảnh có xe → toàn cảnh "trông giống xe" → score cao → đi tiếp
        #    - Sliding window chạy bình thường → phát hiện đúng
        # ══════════════════════════════════════════════════════════════════
        if cfg.get("use_global_gate", True):
            global_score = self._global_gate(gray)
            if global_score < cfg.get("global_reject_thresh", 0.40):
                return []

        # ── Image Pyramid + Sliding Window ────────────────────────────────
        raw_boxes  = []
        raw_scores = []

        win_w, win_h   = cfg["win_size"]
        orig_h, orig_w = gray.shape[:2]

        for level_img, cur_scale in image_pyramid(
                gray,
                scale=cfg["scale_factor"],
                min_size=cfg["min_wh"]):

            for x, y, window in sliding_window(
                    level_img,
                    win_size=(win_w, win_h),
                    step=cfg["step_size"]):

                if window.shape != (win_h, win_w):
                    window = cv2.resize(window, (win_w, win_h))

                feat = extract_hog(window)
                feat = self.scaler.transform(feat.reshape(1, -1))

                if self._has_proba:
                    score = float(self.clf.predict_proba(feat)[0][1])
                else:
                    d     = self.clf.decision_function(feat)[0]
                    score = float(1 / (1 + np.exp(-d)))

                if score < cfg["conf_thresh"]:
                    continue

                # Chuyển tọa độ về ảnh gốc
                x1 = max(0, min(int(x * cur_scale), orig_w - 1))
                y1 = max(0, min(int(y * cur_scale), orig_h - 1))
                x2 = max(0, min(int((x + win_w) * cur_scale), orig_w - 1))
                y2 = max(0, min(int((y + win_h) * cur_scale), orig_h - 1))

                if x2 > x1 and y2 > y1:
                    raw_boxes.append([x1, y1, x2, y2])
                    raw_scores.append(score)

        if not raw_boxes:
            return []

        boxes_arr  = np.array(raw_boxes,  dtype=np.float32)
        scores_arr = np.array(raw_scores, dtype=np.float32)

        if return_all:
            return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s))
                    for b, s in zip(boxes_arr, scores_arr)]

        # ── NMS + Merge boxes ─────────────────────────────────────────────
        results = merge_boxes(
            boxes_arr, scores_arr,
            nms_threshold   = cfg["nms_thresh"],
            merge_threshold = cfg.get("merge_thresh", 0.05),
        )
        return remove_nested_boxes(results)

    # ── Vẽ bbox lên ảnh ───────────────────────────────────────────────────
    @staticmethod
    def draw_boxes(frame: np.ndarray,
                   boxes: List[BBox],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2,
                   show_score: bool = True) -> np.ndarray:
        output = frame.copy()
        for x1, y1, x2, y2, score in boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            if show_score:
                cv2.putText(output, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return output

    @staticmethod
    def get_centroids(boxes: List[BBox]) -> List[Tuple[int, int]]:
        """Tính tọa độ tâm của mỗi bounding box."""
        return [(int((x1 + x2) / 2), int((y1 + y2) / 2))
                for x1, y1, x2, y2, _ in boxes]