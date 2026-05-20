import cv2
import joblib
import numpy as np

from scipy.ndimage import label as scipy_label

from config import *

model_full = joblib.load(MODEL_PATH)

scaler = model_full.named_steps["scaler"]
clf    = model_full.named_steps["clf"]

HOG_CV = cv2.HOGDescriptor(
    _winSize=(64,64),
    _blockSize=(16,16),
    _blockStride=(8,8),
    _cellSize=(8,8),
    _nbins=9
)

class DetectionQueue:

    def __init__(self, shape, max_size=10,
                 heat_thresh=7,
                 min_area=150):

        self.shape       = shape[:2]
        self.max_size    = max_size
        self.heat_thresh = heat_thresh
        self.min_area    = min_area

        import queue
        self.queue = queue.Queue(max_size)

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

        labeled, n = scipy_label(binary)

        boxes = []

        for k in range(1, n+1):

            nz = (labeled == k).nonzero()

            if len(nz[0]) < self.min_area:
                continue

            y1,y2 = int(nz[0].min()), int(nz[0].max())
            x1,x2 = int(nz[1].min()), int(nz[1].max())

            boxes.append((x1,y1,x2,y2))

        return boxes
    
def nms(boxes, scores, thr=0.4):

    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores)

    x1,y1,x2,y2 = b[:,0],b[:,1],b[:,2],b[:,3]

    areas = (x2-x1+1)*(y2-y1+1)

    order = s.argsort()[::-1]

    keep = []

    while order.size:

        i = order[0]

        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])

        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(iou <= thr)[0] + 1]

    return keep

def detect(frame, threshold=None):

    thr = threshold if threshold is not None else THRESHOLD

    H,W = frame.shape[:2]

    y1o,y2o = int(H*ROI_TOP), int(H*ROI_BOTTOM)

    roi = frame[y1o:y2o]

    all_feats  = []
    all_coords = []

    for scale in SCALES:

        rh,rw = roi.shape[:2]

        nw,nh = int(rw/scale), int(rh/scale)

        if nw < WIN[0] or nh < WIN[1]:
            continue

        sroi = cv2.resize(roi, (nw,nh))

        ncx = nw // 8
        ncy = nh // 8

        cwc = WIN[0] // 8

        for fy in range(0, ncy-cwc, STEP_CELLS):

            for fx in range(0, ncx-cwc, STEP_CELLS):

                px,py = fx*8, fy*8

                patch = sroi[py:py+WIN[1], px:px+WIN[0]]

                if patch.shape[:2] != (WIN[1],WIN[0]):
                    continue

                feat = HOG_CV.compute(patch).flatten()

                all_feats.append(feat)

                all_coords.append((px, py, scale, y1o))

    if not all_feats:
        return []

    feats_arr = np.array(all_feats)

    feats_sc = scaler.transform(feats_arr)

    probs_batch = clf.predict_proba(feats_sc)[:,1]

    all_boxes  = []
    all_scores = []

    for i,(px,py,scale,y1o) in enumerate(all_coords):

        if probs_batch[i] >= thr:

            all_boxes.append([
                int(px*scale),
                int(py*scale)+y1o,
                int((px+WIN[0])*scale),
                int((py+WIN[1])*scale)+y1o
            ])

            all_scores.append(probs_batch[i])

    keep = nms(all_boxes, all_scores)

    return [all_boxes[i] for i in keep]