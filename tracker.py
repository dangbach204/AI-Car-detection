"""
tracker.py
==========
Theo dõi xe giữa các frame và đếm số lượng xe.

Thuật toán Centroid Tracker:
    1. Mỗi xe được gán một ID duy nhất khi xuất hiện lần đầu.
    2. Giữa 2 frame liên tiếp, khớp các tâm mới với tâm cũ bằng
       khoảng cách Euclidean (Hungarian algorithm / greedy).
    3. Nếu không khớp → tạo object mới.
    4. Nếu object mất tích quá nhiều frame → xóa khỏi danh sách.
    5. Khi tâm xe đi qua đường đếm (counting line) → tăng bộ đếm.

Đường đếm (Counting Line):
    - Một đường ngang hoặc dọc cố định trên frame.
    - Khi tâm của xe đi từ phía này sang phía kia → đếm.
    - Hướng di chuyển (vào/ra) được phân biệt để tính 2 chiều.
"""

import numpy as np
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Optional


# ── Kiểu dữ liệu ────────────────────────────────────────────────────────────
Centroid = Tuple[int, int]            # (cx, cy)
ObjectID  = int


# ═══════════════════════════════════════════════════════════════════════════
#  CENTROID TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class CentroidTracker:
    """
    Theo dõi các đối tượng chuyển động bằng thuật toán centroid.

    Attributes
    ----------
    next_id      : int    — ID tiếp theo sẽ được gán
    objects      : dict   — {id: centroid} — đối tượng đang theo dõi
    disappeared  : dict   — {id: n_frames} — số frame biến mất
    trajectories : dict   — {id: [centroid, ...]} — lịch sử tọa độ
    """

    def __init__(self,
                 max_disappeared: int = 10,
                 max_distance: int = 80):
        """
        Parameters
        ----------
        max_disappeared : int
            Số frame tối đa một object có thể vắng mặt trước khi bị xóa.
        max_distance    : int
            Khoảng cách pixel tối đa để khớp 2 centroid giữa 2 frame.
        """
        self.next_id       : ObjectID = 0
        self.objects       : OrderedDict[ObjectID, Centroid] = OrderedDict()
        self.disappeared   : Dict[ObjectID, int] = {}
        self.trajectories  : Dict[ObjectID, List[Centroid]] = defaultdict(list)
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    # ── Đăng ký đối tượng mới ──────────────────────────────────────────────
    def _register(self, centroid: Centroid):
        oid = self.next_id
        self.objects[oid]     = centroid
        self.disappeared[oid] = 0
        self.trajectories[oid].append(centroid)
        self.next_id += 1
        return oid

    # ── Xóa đối tượng đã biến mất quá lâu ─────────────────────────────────
    def _deregister(self, oid: ObjectID):
        del self.objects[oid]
        del self.disappeared[oid]

    # ── Cập nhật danh sách đối tượng ──────────────────────────────────────
    def update(self, input_centroids: List[Centroid]) -> OrderedDict:
        """
        Cập nhật tracker với danh sách centroid từ frame mới.

        Parameters
        ----------
        input_centroids : List[(cx, cy)]
            Danh sách tâm bbox phát hiện được trong frame hiện tại.

        Returns
        -------
        OrderedDict {id: centroid}
            Danh sách đối tượng đang được theo dõi.
        """

        # ── Không có bbox → mọi object đều "biến mất" ────────────────────
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        # ── Chưa có object nào → đăng ký tất cả ──────────────────────────
        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
            return self.objects

        # ── Khớp centroid cũ ↔ centroid mới (greedy matching) ────────────
        old_ids       = list(self.objects.keys())
        old_centroids = list(self.objects.values())
        input_centroids_arr = np.array(input_centroids, dtype=np.float32)
        old_centroids_arr   = np.array(old_centroids,   dtype=np.float32)

        # Ma trận khoảng cách Euclidean: D[i, j] = dist(old[i], new[j])
        D = np.linalg.norm(
            old_centroids_arr[:, np.newaxis] - input_centroids_arr[np.newaxis, :],
            axis=2
        )  # shape: (n_old, n_new)

        # Sắp xếp theo hàng (old) có giá trị min ở mỗi cột (new)
        row_mins    = D.min(axis=1).argsort()
        col_of_mins = D.argmin(axis=1)[row_mins]

        used_rows, used_cols = set(), set()
        matched_oids = {}

        for row, col in zip(row_mins, col_of_mins):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            oid = old_ids[row]
            new_c = input_centroids[col]
            self.objects[oid]   = new_c
            self.disappeared[oid] = 0
            self.trajectories[oid].append(new_c)
            matched_oids[oid]   = new_c
            used_rows.add(row)
            used_cols.add(col)

        # Xử lý các old object không khớp
        unmatched_rows = set(range(len(old_ids))) - used_rows
        for row in unmatched_rows:
            oid = old_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        # Đăng ký các new centroid chưa khớp
        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self._register(input_centroids[col])

        return self.objects

    def get_trajectory(self, oid: ObjectID) -> List[Centroid]:
        """Lấy lịch sử tọa độ của một object."""
        return self.trajectories.get(oid, [])


# ═══════════════════════════════════════════════════════════════════════════
#  COUNTING LINE
# ═══════════════════════════════════════════════════════════════════════════

class CountingLine:
    """
    Đường đếm xe.

    Khi tâm của một xe đi từ một phía sang phía kia của đường này
    trong 2 frame liên tiếp → tăng bộ đếm.

    Hỗ trợ:
    - Đường ngang (horizontal): y = const
    - Đường dọc  (vertical)  : x = const
    - Đếm 2 chiều (in/out)
    """

    def __init__(self,
                 position: int,
                 orientation: str = "horizontal",
                 margin: int = 5):
        """
        Parameters
        ----------
        position    : int    — vị trí đường (y_pixel nếu ngang, x_pixel nếu dọc)
        orientation : str    — "horizontal" hoặc "vertical"
        margin      : int    — vùng đệm quanh đường (px) để tránh đếm nhảy
        """
        assert orientation in ("horizontal", "vertical"), \
            "orientation phải là 'horizontal' hoặc 'vertical'"

        self.position    = position
        self.orientation = orientation
        self.margin      = margin

        # Trạng thái: lưu vị trí trước của mỗi object
        self._prev_side  : Dict[ObjectID, int] = {}  # {oid: side}
        self.count_in    : int = 0   # đếm chiều vào
        self.count_out   : int = 0   # đếm chiều ra
        self.counted_ids : set = set()  # tập ID đã được đếm

    @property
    def total(self) -> int:
        return self.count_in + self.count_out

    def _get_side(self, centroid: Centroid) -> int:
        """
        Xác định centroid ở bên nào của đường.

        Returns: -1 (trước đường), +1 (sau đường), 0 (trên đường)
        """
        value = centroid[1] if self.orientation == "horizontal" else centroid[0]
        if value < self.position - self.margin:
            return -1
        elif value > self.position + self.margin:
            return +1
        else:
            return 0   # trong vùng đệm

    def update(self, objects: Dict[ObjectID, Centroid]):
        """
        Kiểm tra các object vừa đi qua đường đếm.

        Parameters
        ----------
        objects : Dict[oid, (cx, cy)]
            Danh sách đối tượng hiện tại từ CentroidTracker.

        Returns
        -------
        List[(oid, direction)] — danh sách xe vừa được đếm trong frame này
        """
        new_counts = []

        for oid, centroid in objects.items():
            current_side = self._get_side(centroid)

            if oid not in self._prev_side:
                # Lần đầu gặp object này
                if current_side != 0:
                    self._prev_side[oid] = current_side
                continue

            prev_side = self._prev_side[oid]

            # Bỏ qua nếu đang trong vùng đệm
            if current_side == 0 or prev_side == 0:
                continue

            # Phát hiện đi qua đường
            if prev_side != current_side and oid not in self.counted_ids:
                direction = "in" if current_side == 1 else "out"
                if direction == "in":
                    self.count_in  += 1
                else:
                    self.count_out += 1
                self.counted_ids.add(oid)
                new_counts.append((oid, direction))

            self._prev_side[oid] = current_side

        return new_counts

    def draw(self,
             frame: np.ndarray,
             color_normal: Tuple[int, int, int]  = (255, 255, 0),
             color_active: Tuple[int, int, int]  = (0, 0, 255),
             thickness: int = 2,
             label: bool = True) -> np.ndarray:
        """
        Vẽ đường đếm và thông tin lên ảnh.

        Parameters
        ----------
        frame        : np.ndarray (BGR)
        color_normal : màu đường bình thường
        color_active : màu đường khi vừa có xe qua (không dùng ở đây, truyền từ ngoài)
        thickness    : độ dày
        label        : hiển thị nhãn count

        Returns
        -------
        np.ndarray
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        if self.orientation == "horizontal":
            cv2.line(output,
                     (0, self.position), (w, self.position),
                     color_normal, thickness)
            if label:
                text = f"IN:{self.count_in}  OUT:{self.count_out}"
                cv2.putText(output, text,
                            (10, self.position - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            color_normal, 2, cv2.LINE_AA)
        else:
            cv2.line(output,
                     (self.position, 0), (self.position, h),
                     color_normal, thickness)
            if label:
                text = f"IN:{self.count_in} OUT:{self.count_out}"
                cv2.putText(output, text,
                            (self.position + 5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            color_normal, 2, cv2.LINE_AA)
        return output

    def reset(self):
        """Đặt lại bộ đếm."""
        self._prev_side.clear()
        self.counted_ids.clear()
        self.count_in  = 0
        self.count_out = 0


import cv2  # noqa: E402 (import ở cuối để tránh circular)


# ═══════════════════════════════════════════════════════════════════════════
#  VEHICLE COUNTER (kết hợp Tracker + CountingLine)
# ═══════════════════════════════════════════════════════════════════════════

class VehicleCounter:
    """
    Kết hợp CentroidTracker và CountingLine để đếm xe.

    Sử dụng:
        counter = VehicleCounter(line_y=300)
        for frame, boxes in video_stream:
            centroids = VehicleDetector.get_centroids(boxes)
            count, objects = counter.update(centroids)
            annotated = counter.draw(frame, boxes, objects)
    """

    def __init__(self,
                 line_position: int,
                 orientation: str = "horizontal",
                 max_disappeared: int = 10,
                 max_distance: int = 80,
                 line_margin: int = 5):
        """
        Parameters
        ----------
        line_position   : int   — vị trí đường đếm (pixel)
        orientation     : str   — "horizontal" hoặc "vertical"
        max_disappeared : int   — ngưỡng frame vắng mặt trước khi xóa object
        max_distance    : int   — khoảng cách tối đa để khớp centroid
        line_margin     : int   — vùng đệm quanh đường đếm
        """
        self.tracker = CentroidTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance
        )
        self.line = CountingLine(
            position=line_position,
            orientation=orientation,
            margin=line_margin
        )
        self._flash_ids: set = set()   # ID vừa đi qua để hiệu ứng flash

    def update(self, centroids: List[Centroid]):
        """
        Cập nhật tracker và kiểm tra đường đếm.

        Parameters
        ----------
        centroids : List[(cx, cy)]

        Returns
        -------
        objects    : OrderedDict {oid: centroid}
        new_counts : List[(oid, direction)]  — xe vừa được đếm frame này
        """
        objects    = self.tracker.update(centroids)
        new_counts = self.line.update(objects)

        self._flash_ids = {oid for oid, _ in new_counts}
        return objects, new_counts

    def draw(self,
             frame: np.ndarray,
             boxes: list,
             objects: dict,
             show_id: bool = True,
             show_trajectory: bool = False) -> np.ndarray:
        """
        Vẽ toàn bộ thông tin lên frame.

        Parameters
        ----------
        frame            : np.ndarray (BGR)
        boxes            : danh sách bbox [(x1,y1,x2,y2,score), ...]
        objects          : dict {oid: centroid}
        show_id          : hiển thị ID xe
        show_trajectory  : hiển thị quỹ đạo di chuyển

        Returns
        -------
        np.ndarray — frame đã vẽ
        """
        output = frame.copy()
        h, w   = frame.shape[:2]

        # ── Màu đường đếm ─────────────────────────────────────────────────
        line_color = (0, 0, 255) if self._flash_ids else (0, 255, 255)
        output = self.line.draw(output, color_normal=line_color)

        # ── Bbox ──────────────────────────────────────────────────────────
        for x1, y1, x2, y2, score in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Kiểm tra xem bbox này có oid nào không
            matched_oid = None
            for oid, centroid in objects.items():
                if abs(centroid[0] - cx) < 10 and abs(centroid[1] - cy) < 10:
                    matched_oid = oid
                    break

            flash = matched_oid in self._flash_ids
            box_color = (0, 0, 255) if flash else (0, 200, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

            if matched_oid is not None and show_id:
                cv2.putText(output, f"#{matched_oid}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            box_color, 1, cv2.LINE_AA)

        # ── Quỹ đạo ───────────────────────────────────────────────────────
        if show_trajectory:
            for oid, traj in self.tracker.trajectories.items():
                if len(traj) < 2:
                    continue
                pts = np.array(traj[-20:], dtype=np.int32)  # 20 điểm gần nhất
                for i in range(1, len(pts)):
                    cv2.line(output, tuple(pts[i-1]), tuple(pts[i]),
                             (200, 200, 0), 1)

        # ── HUD (Heads-Up Display) ────────────────────────────────────────
        # Vẽ nền mờ
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (220, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

        cv2.putText(output, f"Tong xe: {self.line.total}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"Vao: {self.line.count_in}  Ra: {self.line.count_out}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(output, f"Dang theo doi: {len(objects)}",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)

        return output

    @property
    def total_count(self) -> int:
        return self.line.total

    @property
    def count_in(self) -> int:
        return self.line.count_in

    @property
    def count_out(self) -> int:
        return self.line.count_out

    def reset(self):
        """Reset toàn bộ tracker và đường đếm."""
        self.tracker = CentroidTracker(
            max_disappeared=self.tracker.max_disappeared,
            max_distance=self.tracker.max_distance
        )
        self.line.reset()
        self._flash_ids.clear()
