import os
import cv2
import time
import base64
import tempfile
import numpy as np

from detector import detect, DetectionQueue

from flask import (
    Response,
    render_template,
    jsonify,
    request
)

from state import state
from config import *
from camera import scan_cameras


# ──────────────────────────────────────────────────────────────
#  Helpers: merge boxes + full pipeline (dùng chung camera/video)
# ──────────────────────────────────────────────────────────────
def merge_overlapping_boxes(boxes, iou_thr=0.25, contain_thr=0.80):
    """Gộp các box chồng lấn thành 1 box union, lặp cho đến ổn định."""
    if len(boxes) <= 1:
        return [tuple(b) for b in boxes]

    bxs = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        n = len(bxs)
        used = [False] * n
        merged = []
        for i in range(n):
            if used[i]:
                continue
            x1, y1, x2, y2 = bxs[i]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = bxs[j]
                ix1, iy1 = max(x1, bx1), max(y1, by1)
                ix2, iy2 = min(x2, bx2), min(y2, by2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                if inter == 0:
                    continue
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (bx2 - bx1) * (by2 - by1)
                iou = inter / (area_a + area_b - inter)
                contain = inter / min(area_a, area_b)
                if iou >= iou_thr or contain >= contain_thr:
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append([x1, y1, x2, y2])
        bxs = merged
    return [tuple(b) for b in bxs]


def _ar_ok(b):
    w, h = b[2] - b[0], b[3] - b[1]
    if h == 0:
        return False
    r = w / h
    return 0.5 <= r <= 2.2


def run_pipeline(frame):
    """Detect + filter + merge + track + count.
    Mutates state["count"] và state["tracks"].
    Returns: (final_boxes, LINE_Y)
    """
    H, W = frame.shape[:2]
    LINE_Y = int(H * LINE_RATIO)

    # Detect raw
    boxes = detect(frame)

    # Filter size
    boxes = [b for b in boxes if (b[2]-b[0]) >= 30 and (b[3]-b[1]) >= 30]
    # Filter aspect ratio
    boxes = [b for b in boxes if _ar_ok(b)]
    # Merge overlapping
    boxes = merge_overlapping_boxes(boxes, iou_thr=0.25, contain_thr=0.80)
    # Filter mega-box
    max_area = (W * H) * 0.40
    boxes = [b for b in boxes if (b[2]-b[0]) * (b[3]-b[1]) <= max_area]

    # Tracking + counting
    TRACKS  = state["tracks"]
    n_old   = len(TRACKS)
    matched = [False] * n_old

    for b in boxes:
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        by     = b[3]

        best, best_d = -1, 9999
        for ti in range(n_old):
            if matched[ti]:
                continue
            t = TRACKS[ti]
            d = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
            if d < 150 and d < best_d:
                best, best_d = ti, d

        if best >= 0:
            t = TRACKS[best]
            t["hits"] = t.get("hits", 1) + 1
            if (t["hits"] >= 2
                    and (not t["counted"])
                    and t["by"] < LINE_Y
                    and by >= LINE_Y):
                state["count"] += 1
                t["counted"] = True
            t["cx"], t["cy"], t["by"], t["missed"] = cx, cy, by, 0
            matched[best] = True
        else:
            TRACKS.append({
                "cx": cx, "cy": cy, "by": by,
                "counted": False, "missed": 0, "hits": 1
            })

    for ti in range(n_old):
        if not matched[ti]:
            TRACKS[ti]["missed"] += 1
    state["tracks"] = [t for t in TRACKS if t["missed"] < 5]

    return boxes, LINE_Y


# ──────────────────────────────────────────────────────────────
#  MJPEG generator cho video upload mode (2 threads)
#  - Main: đọc + hiển thị frame ở source FPS
#  - Detect thread: chạy run_pipeline trên frame mới nhất, cập nhật
#    box + count khi nào kịp. Không block playback.
# ──────────────────────────────────────────────────────────────
import threading


def gen_video_frames():
    path = state.get("video_path")
    if not path or not os.path.exists(path):
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    target_dt = 1.0 / src_fps      # mục tiêu: phát đúng tốc độ gốc

    enc_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]

    # Shared state giữa main và detect thread
    shared = {
        "latest_frame": None,
        "boxes":        [],
        "line_y":       None,
        "stop":         False,
    }
    lock = threading.Lock()

    def _detect_loop():
        """Chạy nền: lấy frame mới nhất, run_pipeline, cập nhật box."""
        while not shared["stop"]:
            with lock:
                f = shared["latest_frame"]
            if f is None:
                time.sleep(0.01)
                continue
            try:
                bx, ly = run_pipeline(f)
                with lock:
                    shared["boxes"]  = bx
                    shared["line_y"] = ly
            except Exception as e:
                print("[video detect_thread] error:", e)
                time.sleep(0.05)

    det_th = threading.Thread(target=_detect_loop, daemon=True)
    det_th.start()

    n = 0
    t_start = time.time()
    t_fps   = time.time()
    det_n   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            H, W  = frame.shape[:2]

            # Đẩy frame mới nhất cho detect thread + lấy box hiện có
            with lock:
                shared["latest_frame"] = frame.copy()
                boxes  = list(shared["boxes"])
                line_y = shared["line_y"]

            LINE_Y = line_y if line_y is not None else int(H * LINE_RATIO)

            # Vẽ box mới nhất từ detect thread (có thể là từ 1-3 frame trước)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
                cv2.putText(frame, "Vehicle", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1)

            cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)

            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (170, 36), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, f"Count: {state['count']}", (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            _, buf    = cv2.imencode(".jpg", frame, enc_params)
            buf_bytes = buf.tobytes()

            n += 1
            if n % 10 == 0:
                state["fps_stream"] = round(10 / (time.time() - t_fps), 1)
                t_fps = time.time()

            # ★ Khóa tốc độ phát = source FPS, bất kể detect nhanh/chậm
            elapsed  = time.time() - t_start
            expected = n * target_dt
            if elapsed < expected:
                time.sleep(expected - elapsed)

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(buf_bytes)).encode() + b"\r\n"
                   b"\r\n" + buf_bytes + b"\r\n")
    finally:
        shared["stop"] = True
        det_th.join(timeout=1.0)
        cap.release()
        print(f"[video_stream] finished, total frames = {n}")


# ──────────────────────────────────────────────────────────────
#  Generator camera (giữ nguyên, không dùng trong mobile flow)
# ──────────────────────────────────────────────────────────────
def gen_frames():
    t0          = time.time()
    n           = 0
    enc_params  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]
    COLORS      = [(0,255,0),(255,0,255),(0,165,255),(255,255,0),(0,255,255)]
    TARGET_FPS  = 20
    FRAME_DELAY = 1.0 / TARGET_FPS
    last_sent   = time.time()
    while True:
        if not state["cam_on"]:
            time.sleep(0.05); continue

        now  = time.time()
        wait = FRAME_DELAY - (now - last_sent)
        if wait > 0:
            time.sleep(wait)
        last_sent = time.time()

        with state["lock_raw"]:
            frame = state["raw_frame"]
        if frame is None:
            time.sleep(0.005); continue

        vis    = frame.copy()
        H, W   = vis.shape[:2]

        with state["lock_annot"]:
            boxes      = state["last_boxes"]
            line_y_val = state["line_y"]
            LINE_Y     = line_y_val if line_y_val > 0 else int(H * LINE_RATIO)

        for i,(x1,y1,x2,y2) in enumerate(boxes):
            col = COLORS[i % len(COLORS)]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            cv2.circle(vis,((x1+x2)//2,(y1+y2)//2),5,col,-1)

        cv2.line(vis,(0,LINE_Y),(W,LINE_Y),(0,0,255),2)

        ov = vis.copy()
        cv2.rectangle(ov,(0,0),(160,36),(0,0,0),-1)
        cv2.addWeighted(ov,0.4,vis,0.6,0,vis)
        cv2.putText(vis, "Count: " + str(state["count"]),
                    (6,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        _, buf    = cv2.imencode(".jpg", vis, enc_params)
        buf_bytes = buf.tobytes()
        n += 1
        if n % 20 == 0:
            state["fps_stream"] = round(20/(time.time()-t0), 1)
            t0 = time.time()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(buf_bytes)).encode() + b"\r\n"
               b"\r\n" + buf_bytes + b"\r\n")


# ──────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────
def register_routes(app):

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/cameras")
    def cameras():
        return jsonify({"cameras": scan_cameras()})

    @app.route("/video")
    def video():
        resp = Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"]     = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"]            = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.route("/start_camera")
    def start_camera():
        idx_raw = request.args.get("index", "0")
        if idx_raw == "":
            return jsonify({"ok": False, "error": "No camera selected"})
        idx = int(idx_raw)
        with state["cap_lock"]:
            if state["cap"] is not None:
                state["cap"].release()
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                return jsonify({"ok": False})
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            cap.set(cv2.CAP_PROP_FPS,          30)
            state["cap"]         = cap
            state["cam_index"]   = idx
            state["cam_on"]      = True
            state["count"]       = 0
            state["annot_frame"] = None
        return jsonify({"ok": True})

    @app.route("/stop_camera")
    def stop_camera():
        with state["cap_lock"]:
            state["cam_on"] = False
            if state["cap"] is not None:
                state["cap"].release()
                state["cap"] = None
            state["raw_frame"]   = None
            state["annot_frame"] = None
        return jsonify({"ok": True})

    @app.route("/stats")
    def stats():
        return jsonify({
            "count":      state["count"],
            "fps_detect": state["fps_detect"],
            "fps_stream": state["fps_stream"]
        })

    @app.route("/reset")
    def reset():
        last = state.get("count", 0)
        state["reset_flag"]   = True
        with state["lock_annot"]:
            state["last_boxes"] = []
        state["count"]        = 0
        state["tracks"]       = []
        state["detect_queue"] = None
        state["queue_shape"]  = None
        return jsonify({"ok": True, "last_count": last})

    # ── Camera (mobile) detection endpoint ─────────────────────
    @app.route("/detect_frame", methods=["POST"])
    def detect_frame():
        data        = request.json
        image_data  = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr      = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (640, 480))

        H, W = frame.shape[:2]
        boxes, LINE_Y = run_pipeline(frame)

        print(f"[detect_frame] {len(boxes)} boxes, count={state['count']}")

        result = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes]
        return jsonify({
            "count":   state["count"],
            "boxes":   result,
            "line_y":  LINE_Y,
            "frame_w": W,
            "frame_h": H
        })

    # ── Video upload mode ──────────────────────────────────────
    @app.route("/upload_video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"ok": False, "error": "No file in 'video' field"})

        f = request.files["video"]
        if not f.filename:
            return jsonify({"ok": False, "error": "Empty filename"})

        # Save vào thư mục tạm
        tmp_dir   = tempfile.gettempdir()
        ext       = os.path.splitext(f.filename)[1] or ".mp4"
        save_path = os.path.join(tmp_dir, "uploaded_video" + ext)
        f.save(save_path)

        # Reset state cho phiên mới
        state["count"]       = 0
        state["tracks"]      = []
        state["video_path"]  = save_path
        state["reset_flag"]  = True

        print(f"[upload_video] saved to {save_path}, size={os.path.getsize(save_path)} bytes")

        return jsonify({"ok": True, "filename": f.filename})

    @app.route("/video_stream")
    def video_stream():
        resp = Response(
            gen_video_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
        resp.headers["Cache-Control"]     = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"]            = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp