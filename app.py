import cv2, joblib, numpy as np, threading, time, os, base64
import queue as _queue_module
from flask import Flask, Response, render_template_string, request, jsonify
from scipy.ndimage import label as scipy_label

MODEL_PATH   = "vehicle_svm_v3.pkl"
WIN          = (64, 64)
LINE_RATIO   = 0.65
LINE_MARGIN  = 20
THRESHOLD    = 0.5
SCALES       = (1.0, 1.5)
STEP_CELLS   = 3
ROI_TOP      = 0.4
ROI_BOTTOM   = 0.92
CAP_W        = 320
CAP_H        = 240
JPEG_Q       = 60
DETECT_EVERY = 3
HEAT_DECAY   = 0.92
HEAT_THRESH  = 1.5

model_full = joblib.load(MODEL_PATH)
scaler     = model_full.named_steps["scaler"]
clf        = model_full.named_steps["clf"]

HOG_CV = cv2.HOGDescriptor(
    _winSize=(64,64), _blockSize=(16,16),
    _blockStride=(8,8), _cellSize=(8,8), _nbins=9
)

class Heatmap:
    def __init__(self, shape):
        self.map = np.zeros(shape[:2], dtype=np.float32)
    def update(self, boxes):
        self.map *= HEAT_DECAY
        for b in boxes:
            self.map[b[1]:b[3], b[0]:b[2]] += 1.5
    def get_boxes(self):
        binary = (self.map >= HEAT_THRESH).astype(np.uint8)
        labeled, n = scipy_label(binary)
        boxes = []
        for k in range(1, n+1):
            nz = (labeled==k).nonzero()
            if len(nz[0]) < 150: continue
            y1,y2 = int(nz[0].min()),int(nz[0].max())
            x1,x2 = int(nz[1].min()),int(nz[1].max())
            boxes.append((x1,y1,x2,y2))
        return boxes
class DetectionQueue:
    def __init__(self, shape, max_size=10, heat_thresh=7, min_area=150):
        self.shape       = shape[:2]
        self.max_size    = max_size
        self.heat_thresh = heat_thresh
        self.min_area    = min_area
        self.queue       = _queue_module.Queue(max_size)
    def update(self, boxes):
        if self.queue.qsize() == self.max_size:
            self.queue.get()
        self.queue.put(boxes)
    def get_boxes(self):
        heatmap = np.zeros(self.shape, dtype=np.float32)
        for boxes in list(self.queue.queue):
            for b in boxes:
                heatmap[b[1]:b[3], b[0]:b[2]] += 1
        heatmap[heatmap <= self.heat_thresh] = 0
        binary = (heatmap > 0).astype(np.uint8)
        from scipy.ndimage import label as _label
        labeled, n = _label(binary)
        boxes = []
        for k in range(1, n+1):
            nz = (labeled==k).nonzero()
            if len(nz[0]) < self.min_area: continue
            y1,y2 = int(nz[0].min()),int(nz[0].max())
            x1,x2 = int(nz[1].min()),int(nz[1].max())
            boxes.append((x1,y1,x2,y2))
        return boxes

def nms(boxes, scores, thr=0.4):
    if not boxes: return []
    b=np.array(boxes,dtype=np.float32); s=np.array(scores)
    x1,y1,x2,y2=b[:,0],b[:,1],b[:,2],b[:,3]
    areas=(x2-x1+1)*(y2-y1+1); order=s.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1+1)*np.maximum(0,yy2-yy1+1)
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[np.where(iou<=thr)[0]+1]
    return keep

def detect(frame, threshold=None):
    thr = threshold if threshold is not None else THRESHOLD
    H,W = frame.shape[:2]
    y1o,y2o = int(H*ROI_TOP),int(H*ROI_BOTTOM)
    roi = frame[y1o:y2o]
    all_feats, all_coords = [], []
    for scale in SCALES:
        rh,rw = roi.shape[:2]
        nw,nh = int(rw/scale),int(rh/scale)
        if nw<WIN[0] or nh<WIN[1]: continue
        sroi = cv2.resize(roi,(nw,nh))
        ncx = nw//8; ncy = nh//8; cwc = WIN[0]//8
        for fy in range(0, ncy-cwc, STEP_CELLS):
            for fx in range(0, ncx-cwc, STEP_CELLS):
                px,py = fx*8, fy*8
                patch = sroi[py:py+WIN[1], px:px+WIN[0]]
                if patch.shape[:2]!=(WIN[1],WIN[0]): continue
                all_feats.append(HOG_CV.compute(patch).flatten())
                all_coords.append((px, py, scale, y1o))
    if not all_feats:
        return []
    feats_arr   = np.array(all_feats)
    feats_sc    = scaler.transform(feats_arr)
    probs_batch = clf.predict_proba(feats_sc)[:, 1]
    all_boxes, all_scores = [], []
    for i,(px,py,scale,y1o) in enumerate(all_coords):
        if probs_batch[i] >= thr:
            all_boxes.append([int(px*scale),int(py*scale)+y1o,
                              int((px+WIN[0])*scale),int((py+WIN[1])*scale)+y1o])
            all_scores.append(probs_batch[i])
    keep = nms(all_boxes, all_scores)
    return [all_boxes[i] for i in keep]

# ── Shared state ──────────────────────────────────────────────
state = {
    "raw_frame"  : None,
    "annot_frame": None,
    "count"      : 0,
    "fps_detect" : 0.0,
    "fps_stream" : 0.0,
    "lock_raw"   : threading.Lock(),
    "lock_annot" : threading.Lock(),
    "cam_on"     : False,
    "cam_index"  : 0,
    "cap"        : None,
    "cap_lock"   : threading.Lock(),
    "last_boxes" : [],       # thêm
    "line_y"     : 0,        # thêm
    "frame_hw"   : None,     # thêm
    "reset_flag" : False,   # ← thêm
}

def scan_cameras(max_test=5):
   
    found = []
    # Lấy index của camera đang được bật (nếu có)
    with state["cap_lock"]:
        active_idx = state["cam_index"] if state["cam_on"] else -1
        is_active_opened = state["cap"].isOpened() if state["cap"] else False

    for i in range(max_test):
        # NẾU camera này chính là camera đang được app mở -> Không chọc vào mở lại nữa, báo có luôn.
        if i == active_idx and is_active_opened:
            found.append({"index": i, "name": f"Camera {i} (Đang dùng)"})
            continue
            
        # Nếu là camera khác, thử mở xem có tồn tại không
        # Mẹo: Trên Windows, đôi khi thêm cv2.CAP_DSHOW giúp quét nhanh và tránh treo
        cap = cv2.VideoCapture(i) 
        if cap.isOpened():
            found.append({"index": i, "name": f"Camera {i}"})
            cap.release()
            
    return found

def camera_thread():
    while True:
        with state["cap_lock"]:
            cap = state["cap"]
            cam_on = state["cam_on"]
        if not cam_on or cap is None:
            time.sleep(0.05); continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue
        with state["lock_raw"]:
            state["raw_frame"] = frame

def detect_thread():
    heat          = None
    vehicle_count = 0
    prev_centers  = []
    frame_idx     = 0
    last_shape    = None
    t0            = time.time()
    while True:
        if not state["cam_on"]:
            time.sleep(0.05); continue
        with state["lock_raw"]:
            frame = state["raw_frame"]
        if frame is None:
            time.sleep(0.01); continue
        H,W    = frame.shape[:2]
        LINE_Y = int(H * LINE_RATIO)
        if heat is None or last_shape != (H,W):
            heat          = DetectionQueue((H,W), max_size=QUEUE_SIZE, heat_thresh=QUEUE_THRESH)
            last_shape    = (H,W)
            prev_centers  = []
            vehicle_count = state["count"]
            frame_idx     = 0
            t0            = time.time()
        if frame_idx % DETECT_EVERY == 0:
            raw_boxes = detect(frame)
            heat.update(raw_boxes)
        final_boxes = heat.get_boxes()
        curr_c = [((b[0]+b[2])//2,(b[1]+b[3])//2) for b in final_boxes]
        counted_this_frame = set()
        for cx,cy in curr_c:
            matched_prev_y = None
            min_dist       = float("inf")
            matched_idx    = -1
            for pi,(px,py) in enumerate(prev_centers):
                dist = ((cx-px)**2+(cy-py)**2)**0.5
                max_dist = max(80, H*0.12)
                if dist < max_dist and dist < min_dist:
                    min_dist       = dist
                    matched_prev_y = py
                    matched_idx    = pi
            if (matched_prev_y is not None
                    and matched_prev_y < LINE_Y
                    and cy >= LINE_Y
                    and matched_idx not in counted_this_frame):
                vehicle_count += 1
                counted_this_frame.add(matched_idx)
        prev_centers = curr_c

        # Xử lý reset
        if state["reset_flag"]:
            vehicle_count       = 0
            prev_centers        = []
            heat                = DetectionQueue((H,W), max_size=QUEUE_SIZE, heat_thresh=QUEUE_THRESH)
            state["reset_flag"] = False
            state["count"]      = 0
        else:
            state["count"] = vehicle_count

        frame_idx += 1
        if frame_idx % 10 == 0:
            state["fps_detect"] = round(10/(time.time()-t0), 1)
            t0 = time.time()
        with state["lock_annot"]:
            state["last_boxes"] = final_boxes
            state["line_y"]     = LINE_Y
            state["frame_hw"]   = (H,W)

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

# ── Flask ─────────────────────────────────────────────────────
app = Flask(__name__)
_HTML_INLINE = open("html_app.html", encoding="utf-8").read()

@app.route("/")
def index(): return render_template_string(_HTML_INLINE)

@app.route("/video")
def video():
    resp = Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"   # tắt buffer nếu dùng nginx
    return resp

@app.route("/cameras")
def cameras():
    return jsonify({"cameras": scan_cameras()})

@app.route("/start_camera")
def start_camera():
    idx = int(request.args.get("index", 0))
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
        state["cap"]       = cap
        state["cam_index"] = idx
        state["cam_on"]    = True
        state["count"]     = 0
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

@app.route("/set_camera")
def set_camera():
    idx = int(request.args.get("index", 0))
    with state["cap_lock"]:
        if state["cap"] is not None:
            state["cap"].release()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return jsonify({"ok": False})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        state["cap"]       = cap
        state["cam_index"] = idx
        state["annot_frame"] = None
    return jsonify({"ok": True})

@app.route("/stats")
def stats():
    return jsonify({"count": state["count"],
                    "fps_detect": state["fps_detect"],
                    "fps_stream": state["fps_stream"]})

@app.route("/reset")
def reset():
    last = state["count"]
    state["reset_flag"] = True   # ← thêm
    with state["lock_annot"]:
        state["last_boxes"] = []
    return jsonify({"ok": True, "last_count": last})

if __name__ == "__main__":
    print("Starting threads...")
    threading.Thread(target=camera_thread, daemon=True).start()
    time.sleep(0.3)
    threading.Thread(target=detect_thread, daemon=True).start()
    print("Server: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
