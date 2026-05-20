import threading

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

    "last_boxes" : [],
    "line_y"     : 0,
    "frame_hw"   : None,

    "reset_flag" : False,

    "detect_queue": None,
    "queue_shape"   : None,

    "tracks":  [],   # list of {"cx","cy","by","counted","missed"}
}