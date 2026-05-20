import cv2
import time
import base64
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
        state["reset_flag"] = True
        with state["lock_annot"]:
            state["last_boxes"] = []
        state["count"]        = 0
        state["tracks"]       = []
        state["detect_queue"] = None
        state["queue_shape"]  = None
        return jsonify({"ok": True, "last_count": last})

    @app.route("/detect_frame", methods=["POST"])
    def detect_frame():
        # ── 1. Decode ảnh ──────────────────────────────────────────
        data       = request.json
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr     = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (640, 480))

        H, W   = frame.shape[:2]
        LINE_Y = int(H * LINE_RATIO)

        # ── 2. Detect (1 lần duy nhất) ─────────────────────────────
        raw_boxes = detect(frame)

        # ── 3. Heatmap queue để ổn định box ────────────────────────
        if state["detect_queue"] is None or state["queue_shape"] != (H, W):
            state["detect_queue"] = DetectionQueue(
                (H, W), max_size=QUEUE_SIZE, heat_thresh=QUEUE_THRESH
            )
            state["queue_shape"] = (H, W)

        state["detect_queue"].update(raw_boxes)
        boxes = state["detect_queue"].get_boxes()

        # ── 4. Tracking + đếm xe qua line (1 block duy nhất) ───────
        TRACKS  = state["tracks"]
        matched = [False] * len(TRACKS)

        for b in boxes:
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            by     = b[3]

            best, best_d = -1, 9999
            for ti, t in enumerate(TRACKS):
                if matched[ti]:
                    continue
                d = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
                if d < 100 and d < best_d:
                    best, best_d = ti, d

            if best >= 0:
                t = TRACKS[best]
                # cross line → đếm 1 lần cho mỗi track
                if (not t["counted"]) and t["by"] < LINE_Y and by >= LINE_Y:
                    state["count"] += 1
                    t["counted"] = True
                t["cx"], t["cy"], t["by"], t["missed"] = cx, cy, by, 0
                matched[best] = True
            else:
                TRACKS.append({
                    "cx": cx, "cy": cy, "by": by,
                    "counted": False, "missed": 0
                })

        # tăng missed cho track không match, xoá track cũ
        for ti, t in enumerate(TRACKS):
            if not matched[ti]:
                t["missed"] += 1
        state["tracks"] = [t for t in TRACKS if t["missed"] < 8]

        # ── 5. Trả về client (kèm kích thước frame để scale overlay) ─
        result = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes]

        return jsonify({
            "count":   state["count"],
            "boxes":   result,
            "line_y":  LINE_Y,
            "frame_w": W,
            "frame_h": H
        })