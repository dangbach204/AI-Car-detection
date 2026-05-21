MODEL_PATH   = "models/vehicle_svm_v3.pkl"

WIN          = (64, 64)

LINE_RATIO   = 2.0/3.0
LINE_MARGIN  = 20

# Tăng cao hơn để giảm "mega-box" FP ở vạch đường tại các scale lớn
# (scale 2.5 phủ vùng 160px, dễ bắt nhầm vạch đường + bóng thành xe).
THRESHOLD    = 0.50

# Mở rộng scale: 0.75 cho xe xa nhỏ ~48px, 2.5 cho xe gần to ~160px
SCALES       = (0.75, 1.0, 1.5, 2.0, 2.5)
STEP_CELLS   = 3

# Mở rộng ROI để bắt xe ở mép trên/dưới
ROI_TOP      = 0.05
ROI_BOTTOM   = 0.95

CAP_W        = 640
CAP_H        = 480

JPEG_Q       = 60

DETECT_EVERY = 3

HEAT_DECAY   = 0.92
HEAT_THRESH  = 1.5

QUEUE_SIZE   = 3
QUEUE_THRESH = 2