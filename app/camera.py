import cv2
import time

from state import state

from config import *
from detector import detect, DetectionQueue

def scan_cameras(max_test=5):

    found = []

    with state["cap_lock"]:

        active_idx = state["cam_index"] if state["cam_on"] else -1

        is_active_opened = (
            state["cap"].isOpened()
            if state["cap"]
            else False
        )

    for i in range(max_test):

        if i == active_idx and is_active_opened:

            found.append({
                "index": i,
                "name": f"Camera {i} (Đang dùng)"
            })

            continue

        cap = cv2.VideoCapture(i)

        if cap.isOpened():

            found.append({
                "index": i,
                "name": f"Camera {i}"
            })

            cap.release()

    return found

def camera_thread():

    while True:

        with state["cap_lock"]:

            cap = state["cap"]
            cam_on = state["cam_on"]

        if not cam_on or cap is None:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()

        if not ret:
            time.sleep(0.01)
            continue

        with state["lock_raw"]:
            state["raw_frame"] = frame

def detect_thread():

    heat = None

    vehicle_count = 0

    prev_centers = []
    prev_bottoms = []

    frame_idx = 0

    last_shape = None

    t0 = time.time()

    while True:

        if not state["cam_on"]:
            time.sleep(0.05)
            continue

        with state["lock_raw"]:
            frame = state["raw_frame"]

        if frame is None:
            time.sleep(0.01)
            continue

        H, W = frame.shape[:2]

        LINE_Y = int(H * LINE_RATIO)

        # init queue
        if heat is None or last_shape != (H, W):

            heat = DetectionQueue(
                (H, W),
                max_size=QUEUE_SIZE,
                heat_thresh=QUEUE_THRESH
            )

            last_shape = (H, W)

            prev_centers = []

            vehicle_count = state["count"]

            frame_idx = 0

            t0 = time.time()

        # detect
        if frame_idx % DETECT_EVERY == 0:

            raw_boxes = detect(frame)

            heat.update(raw_boxes)

        final_boxes = heat.get_boxes()

        curr_c = []
        curr_bottoms = []
        for b in final_boxes:
            cx = (b[0]+b[2])//2
            cy = (b[1]+b[3])//2
            curr_c.append((cx, cy))
            curr_bottoms.append(b[3])

        counted_this_frame = set()

        for i, (cx, cy) in enumerate(curr_c):

            min_dist = float("inf")

            matched_idx = -1

            for pi, (px, py) in enumerate(prev_centers):

                dist = ((cx-px)**2 + (cy-py)**2)**0.5

                max_dist = max(80, H*0.12)

                if dist < max_dist and dist < min_dist:

                    min_dist = dist

                    matched_idx = pi

            if matched_idx != -1 and matched_idx not in counted_this_frame:
                prev_bottom = prev_bottoms[matched_idx] if matched_idx < len(prev_bottoms) else None
                curr_bottom = curr_bottoms[i] if i < len(curr_bottoms) else None
                if prev_bottom is not None and curr_bottom is not None:
                    if prev_bottom < LINE_Y and curr_bottom >= LINE_Y:
                        vehicle_count += 1
                        counted_this_frame.add(matched_idx)

        prev_centers = curr_c
        prev_bottoms = curr_bottoms

        # reset
        if state["reset_flag"]:

            vehicle_count = 0

            prev_centers = []
            prev_bottoms = []

            heat = DetectionQueue(
                (H, W),
                max_size=QUEUE_SIZE,
                heat_thresh=QUEUE_THRESH
            )

            state["reset_flag"] = False

            state["count"] = 0

        else:

            state["count"] = vehicle_count

        frame_idx += 1

        # fps
        if frame_idx % 10 == 0:

            state["fps_detect"] = round(
                10 / (time.time() - t0),
                1
            )

            t0 = time.time()

        with state["lock_annot"]:

            state["last_boxes"] = final_boxes

            state["line_y"] = LINE_Y

            state["frame_hw"] = (H, W)