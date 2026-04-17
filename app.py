"""
app.py
======
Ứng dụng Streamlit — Đếm xe bằng HOG + SVM

Tính năng:
    • Tải video hoặc dùng camera trực tiếp
    • Điều chỉnh tham số phát hiện theo thời gian thực
    • Hiển thị kết quả đếm xe (vào/ra)
    • Trực quan hóa pipeline HOG
    • Xuất báo cáo CSV

Chạy:
    streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st
import joblib
import tempfile
import os
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import preprocess
from hog_features import extract_hog
from detector import VehicleDetector, DETECT_CFG
from tracker import VehicleCounter
from generate_demo_data import create_demo_video


# ══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH TRANG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Vehicle Counter — HOG + SVM",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tùy chỉnh ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font & background */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

  .stApp { background: #0a0e1a; font-family: 'Inter', sans-serif; }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1117 50%, #1a2744 100%);
    border: 1px solid #2a3a6e;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .header-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      90deg, transparent, transparent 40px,
      rgba(59,130,246,0.03) 40px, rgba(59,130,246,0.03) 41px
    );
  }
  .header-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -1px;
  }
  .header-sub {
    color: #64748b; font-size: 0.95rem; margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #111827, #1e2a45);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
  .metric-value {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; font-family: 'JetBrains Mono', monospace;
  }
  .metric-label { color: #94a3b8; font-size: 0.8rem; margin-top: 6px; letter-spacing: 2px; text-transform: uppercase; }

  /* Alert flash */
  .flash-alert {
    background: rgba(239,68,68,0.15); border: 1px solid #ef4444;
    border-radius: 8px; padding: 10px 16px; color: #fca5a5;
    font-size: 0.85rem; margin: 8px 0;
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
  }

  /* Step badges */
  .step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-weight: 600; color: white;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 10px;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a3e !important;
  }

  /* Metrics override */
  [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 700; }

  /* Hide Streamlit watermark */
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_cached(model_dir: str):
    """Tải model SVM (cache để không load lại mỗi lần render)."""
    model_path  = Path(model_dir) / "svm_vehicle.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"
    meta_path   = Path(model_dir) / "train_meta.pkl"

    if not model_path.exists():
        return None, None, None

    clf    = joblib.load(str(model_path))
    scaler = joblib.load(str(scaler_path))
    meta   = joblib.load(str(meta_path)) if meta_path.exists() else {}
    return clf, scaler, meta


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Chuyển BGR (OpenCV) → RGB (Streamlit)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_hog_visualization(frame: np.ndarray) -> np.ndarray:
    """Trích xuất và vẽ ảnh HOG của frame (resize trước)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 128))
    _, hog_img = extract_hog(small, visualize=True)

    # Scale up để hiển thị
    hog_display = cv2.resize(
        (hog_img * 255).astype(np.uint8),
        (128, 256),
        interpolation=cv2.INTER_NEAREST
    )
    return cv2.applyColorMap(hog_display, cv2.COLORMAP_VIRIDIS)


def frame_to_bytes(frame_rgb: np.ndarray) -> bytes:
    """Encode frame RGB thành JPEG bytes."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def process_frame(frame: np.ndarray,
                  detector: VehicleDetector,
                  counter: VehicleCounter,
                  show_traj: bool = False) -> tuple:
    """
    Xử lý một frame: phát hiện + tracking + đếm.

    Returns (annotated_frame_rgb, boxes, n_total, n_in, n_out)
    """
    boxes     = detector.detect(frame)
    centroids = VehicleDetector.get_centroids(boxes)
    objects, new_counts = counter.update(centroids)

    annotated = counter.draw(frame, boxes, objects,
                             show_trajectory=show_traj)
    return bgr_to_rgb(annotated), boxes, counter.total_count, counter.count_in, counter.count_out


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p style="color:#60a5fa;font-weight:700;font-size:1.1rem;letter-spacing:1px;">⚙️ CONFIGURATION</p>', unsafe_allow_html=True)
    st.divider()

    # ── Model ──────────────────────────────────────────────────────────────
    st.markdown("**📁 Model**")
    model_dir = st.text_input("Thư mục model", value="models",
                               help="Thư mục chứa svm_vehicle.pkl và scaler.pkl")

    clf, scaler, meta = load_model_cached(model_dir)
    if clf is not None:
        acc = meta.get("accuracy", 0)
        st.success(f"✅ Model đã tải | Accuracy: {acc:.1%}")
        if meta.get("hog_feature_size"):
            st.caption(f"HOG features: {meta['hog_feature_size']} chiều")
    else:
        st.error("❌ Chưa có model. Chạy train.py trước!")
        st.info("Demo mode: Dùng OpenCV HOG detector")

    st.divider()

    # ── Tham số phát hiện ─────────────────────────────────────────────────
    st.markdown("**🔍 Phát hiện**")
    conf_thresh = st.slider("Ngưỡng confidence", 0.3, 0.95, 0.55, 0.05,
                             help="Điểm SVM tối thiểu để coi là xe")
    nms_thresh  = st.slider("Ngưỡng NMS (IoU)", 0.1, 0.6, 0.3, 0.05,
                             help="IoU tối đa giữa 2 bbox trước NMS")
    step_size   = st.slider("Bước sliding window (px)", 4, 32, 8, 4,
                             help="Nhỏ hơn → chính xác hơn nhưng chậm hơn")
    scale_factor = st.slider("Image pyramid scale", 1.1, 2.0, 1.25, 0.05,
                              help="Hệ số thu nhỏ mỗi tầng pyramid")

    st.divider()

    # ── Tracking ──────────────────────────────────────────────────────────
    st.markdown("**🎯 Tracking**")
    max_distance   = st.slider("Khoảng cách max centroid (px)", 20, 200, 80, 10)
    max_disappeared = st.slider("Frame tối đa vắng mặt", 1, 30, 10)
    line_pos_pct   = st.slider("Vị trí đường đếm (%)", 10, 90, 50,
                                help="% chiều cao (hoặc rộng) của frame")
    orientation    = st.radio("Hướng đường đếm", ["horizontal", "vertical"],
                               horizontal=True)

    st.divider()

    # ── Hiển thị ──────────────────────────────────────────────────────────
    st.markdown("**🎨 Hiển thị**")
    show_trajectory = st.checkbox("Hiển thị quỹ đạo", value=False)
    show_hog_viz    = st.checkbox("Hiển thị HOG visualization", value=False)
    playback_fps    = st.slider("Tốc độ xử lý (FPS cap)", 1, 60, 15)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="header-title">🚗 Vehicle Counter</div>
  <div class="header-sub">HOG Feature Extraction &nbsp;+&nbsp; SVM Classifier &nbsp;+&nbsp; Centroid Tracker</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_detect, tab_train, tab_explain, tab_about = st.tabs([
    "🎬 Phát hiện & Đếm",
    "🏋️ Huấn luyện",
    "🔬 Pipeline HOG",
    "📖 Hướng dẫn"
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: PHÁT HIỆN & ĐẾM
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:

    col_src, col_opt = st.columns([2, 1])

    with col_src:
        source_mode = st.radio(
            "Nguồn video",
            ["📤 Upload video", "📹 Webcam (thực nghiệm)", "🎬 Video demo"],
            horizontal=True
        )

    video_path = None
    cap = None

    if source_mode == "📤 Upload video":
        uploaded = st.file_uploader("Chọn file video",
                                     type=["mp4", "avi", "mov", "mkv"],
                                     label_visibility="collapsed")
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            video_path = tfile.name

    elif source_mode == "🎬 Video demo":
        demo_path = "data/demo_video.mp4"
        if not Path(demo_path).exists():
            with st.spinner("Đang tạo video demo..."):
                Path("data").mkdir(exist_ok=True)
                create_demo_video(demo_path, n_frames=300)
        video_path = demo_path
        st.info("Đang dùng video demo. Upload video thật để kết quả chính xác hơn.")

    else:  # Webcam
        st.warning("Webcam chỉ hoạt động khi chạy Streamlit local.")
        video_path = 0  # cv2.VideoCapture(0)

    # ── Khởi tạo detector & counter ───────────────────────────────────────
    detect_cfg = {
        **DETECT_CFG,
        "conf_thresh": conf_thresh,
        "nms_thresh":  nms_thresh,
        "step_size":   step_size,
        "scale_factor": scale_factor,
    }

    # ── Nút điều khiển ────────────────────────────────────────────────────
    col_play, col_stop, col_reset = st.columns([1, 1, 1])
    btn_start = col_play.button("▶️ Bắt đầu", type="primary",  use_container_width=True)
    btn_stop  = col_stop.button("⏹ Dừng",   type="secondary", use_container_width=True)

    if "running" not in st.session_state:
        st.session_state.running = False
    if "count_history" not in st.session_state:
        st.session_state.count_history = []

    if btn_start:
        st.session_state.running = True
        st.session_state.count_history = []
    if btn_stop:
        st.session_state.running = False

    # ── Metric cards ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    ph_total = m1.empty()
    ph_in    = m2.empty()
    ph_out   = m3.empty()
    ph_fps   = m4.empty()

    def render_metrics(total, n_in, n_out, fps):
        for ph, val, label, color in [
            (ph_total, total,  "TỔNG XE",   "#60a5fa"),
            (ph_in,    n_in,   "VÀO",       "#34d399"),
            (ph_out,   n_out,  "RA",        "#f87171"),
            (ph_fps,   f"{fps:.0f}", "FPS", "#a78bfa"),
        ]:
            ph.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="background:linear-gradient(90deg,{color},{color}88);-webkit-background-clip:text;">
                {val}
              </div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    render_metrics(0, 0, 0, 0)

    # ── Video output ──────────────────────────────────────────────────────
    col_vid, col_hog = st.columns([3, 1] if show_hog_viz else [1, 0])
    frame_placeholder = col_vid.empty()
    hog_placeholder   = col_hog.empty() if show_hog_viz else None

    # Chart placeholder
    chart_placeholder = st.empty()

    # ── Vòng lặp xử lý video ─────────────────────────────────────────────
    if st.session_state.running and video_path is not None:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_cap      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Vị trí đường đếm theo pixel
        line_pos = int(h * line_pos_pct / 100) if orientation == "horizontal" \
                   else int(w * line_pos_pct / 100)

        # Khởi tạo detector
        if clf is not None:
            detector = VehicleDetector(clf, scaler, cfg=detect_cfg)
        else:
            detector = None  # fallback: chỉ vẽ mà không phát hiện

        counter = VehicleCounter(
            line_position=line_pos,
            orientation=orientation,
            max_disappeared=max_disappeared,
            max_distance=max_distance,
        )

        frame_idx   = 0
        t_start     = time.time()
        delay       = 1.0 / playback_fps

        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t0 = time.time()

            if detector is not None:
                frame_rgb, boxes, total, n_in, n_out = process_frame(
                    frame, detector, counter, show_trajectory=show_trajectory
                )
            else:
                # Fallback: OpenCV HOG người đi bộ → thay bằng xe
                frame_rgb = bgr_to_rgb(frame)
                boxes, total, n_in, n_out = [], counter.total_count, counter.count_in, counter.count_out

            # FPS thực tế
            elapsed = time.time() - t_start
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0

            # Hiển thị frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if show_hog_viz and hog_placeholder:
                hog_vis = draw_hog_visualization(frame)
                hog_placeholder.image(bgr_to_rgb(hog_vis), caption="HOG Visualization",
                                       use_container_width=True)

            # Cập nhật metrics
            render_metrics(total, n_in, n_out, fps_actual)

            # Lưu lịch sử
            st.session_state.count_history.append({
                "frame": frame_idx,
                "time_s": round(frame_idx / fps_cap, 2),
                "total": total,
                "in": n_in,
                "out": n_out,
                "fps": round(fps_actual, 1),
            })

            # Vẽ chart lịch sử
            if len(st.session_state.count_history) > 1:
                df = pd.DataFrame(st.session_state.count_history)
                chart_placeholder.line_chart(
                    df.set_index("time_s")[["total", "in", "out"]],
                    height=200,
                    use_container_width=True,
                )

            # Kiểm soát tốc độ
            proc_time = time.time() - t0
            sleep_time = max(0, delay - proc_time)
            time.sleep(sleep_time)

        cap.release()
        st.session_state.running = False

        # Nút xuất CSV
        if st.session_state.count_history:
            df_result = pd.DataFrame(st.session_state.count_history)
            csv = df_result.to_csv(index=False)
            st.download_button(
                "📥 Tải kết quả CSV",
                data=csv,
                file_name=f"vehicle_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: HUẤN LUYỆN
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("""
    <div class="step-badge">TRAINING PIPELINE</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### Bước 1: Chuẩn bị Dataset

    **Option A — Dùng Kaggle:**
    ```bash
    # Cài Kaggle CLI
    pip install kaggle
    kaggle datasets download -d sshikamaru/car-object-detection
    unzip car-object-detection.zip -d data/
    ```

    **Option B — Tạo dữ liệu mẫu (demo):**
    ```bash
    python src/generate_demo_data.py --n-pos 500 --n-neg 500 --video
    ```
    """)

    col_pos, col_neg = st.columns(2)
    pos_dir_input = col_pos.text_input("📁 Thư mục ảnh CÓ XE",  value="data/positive")
    neg_dir_input = col_neg.text_input("📁 Thư mục ảnh KHÔNG XE", value="data/negative")

    col_c, col_max, col_test = st.columns(3)
    svm_C_val    = col_c.number_input("SVM C", value=0.01, min_value=0.0001, format="%.4f")
    max_imgs     = col_max.number_input("Max ảnh/lớp (0=tất cả)", value=0, min_value=0)
    test_ratio   = col_test.slider("Tỷ lệ test", 0.1, 0.4, 0.2)

    out_dir_input = st.text_input("📦 Thư mục lưu model", value="models")

    if st.button("🚀 Bắt đầu Huấn luyện", type="primary"):
        from train import train as train_fn
        log_area = st.empty()
        prog     = st.progress(0)

        try:
            with st.spinner("Đang tải dữ liệu và trích xuất HOG..."):
                prog.progress(10)
                clf_new, scaler_new, metrics = train_fn(
                    pos_dir=pos_dir_input,
                    neg_dir=neg_dir_input,
                    output_dir=out_dir_input,
                    test_size=test_ratio,
                    max_per_class=int(max_imgs) if max_imgs > 0 else None,
                    svm_C=svm_C_val,
                )
                prog.progress(100)

            st.success("✅ Huấn luyện thành công!")

            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy",   f"{metrics['accuracy']:.2%}")
            m2.metric("Train size", metrics['train_size'])
            m3.metric("Test size",  metrics['test_size'])

            cm = np.array(metrics["confusion_matrix"])
            st.markdown("**Confusion Matrix:**")
            df_cm = pd.DataFrame(cm,
                                  index=["Thực: Không xe", "Thực: Có xe"],
                                  columns=["Dự đoán: Không xe", "Dự đoán: Có xe"])
            st.dataframe(df_cm.style.background_gradient(cmap="Blues"))

            st.info("Reload trang để dùng model mới trong tab Phát hiện.")

        except Exception as e:
            prog.progress(0)
            st.error(f"❌ Lỗi: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: GIẢI THÍCH PIPELINE HOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("""
    <div class="step-badge">PIPELINE VISUALIZATION</div>
    """, unsafe_allow_html=True)

    uploaded_explain = st.file_uploader("Upload ảnh để xem pipeline HOG",
                                         type=["jpg", "png", "jpeg"])

    if uploaded_explain:
        file_bytes = np.frombuffer(uploaded_explain.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.markdown("---")
        st.markdown("### Các bước Pipeline")

        col1, col2, col3, col4 = st.columns(4)

        # Bước 1: Ảnh gốc
        with col1:
            st.markdown('<div class="step-badge">1. Ảnh gốc</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(img_bgr), use_container_width=True)
            h0, w0 = img_bgr.shape[:2]
            st.caption(f"{w0}×{h0} px | BGR")

        # Bước 2: Resize + Grayscale
        with col2:
            st.markdown('<div class="step-badge">2. Resize + Grayscale</div>', unsafe_allow_html=True)
            gray = cv2.cvtColor(cv2.resize(img_bgr, (64, 128)), cv2.COLOR_BGR2GRAY)
            st.image(gray, use_container_width=True, clamp=True)
            st.caption("64×128 px | Grayscale")

        # Bước 3: CLAHE
        with col3:
            st.markdown('<div class="step-badge">3. CLAHE</div>', unsafe_allow_html=True)
            from preprocess import apply_clahe
            clahe_img = apply_clahe(gray)
            st.image(clahe_img, use_container_width=True, clamp=True)
            st.caption("Cân bằng sáng thích nghi")

        # Bước 4: HOG
        with col4:
            st.markdown('<div class="step-badge">4. HOG Features</div>', unsafe_allow_html=True)
            from skimage import exposure as skexp
            from skimage.feature import hog
            feat, hog_img = hog(
                clahe_img.astype(np.float32) / 255.0,
                orientations=9, pixels_per_cell=(8,8),
                cells_per_block=(2,2), visualize=True,
                feature_vector=True
            )
            hog_display = skexp.rescale_intensity(hog_img, in_range=(0, 10))
            hog_color   = cv2.applyColorMap(
                (hog_display * 255).astype(np.uint8),
                cv2.COLORMAP_VIRIDIS
            )
            st.image(bgr_to_rgb(hog_color), use_container_width=True)
            st.caption(f"Vector: {len(feat)} chiều")

        # Vector visualization
        st.markdown("---")
        st.markdown("### Vector đặc trưng HOG (128 giá trị đầu tiên)")
        feat_df = pd.DataFrame({
            "dimension": list(range(128)),
            "value":     feat[:128].tolist()
        })
        st.bar_chart(feat_df.set_index("dimension")["value"], height=200)
        st.caption(f"Tổng {len(feat)} chiều. Min={feat.min():.4f} Max={feat.max():.4f} Mean={feat.mean():.4f}")

    else:
        st.info("Upload một ảnh để xem từng bước của pipeline HOG.")

        # Ảnh mẫu giải thích
        st.markdown("""
        #### HOG — Histogram of Oriented Gradients

        | Tham số | Giá trị | Ý nghĩa |
        |---------|---------|---------|
        | `orientations` | 9 | Số bin hướng gradient (0°–180°) |
        | `pixels_per_cell` | 8×8 | Kích thước cell |
        | `cells_per_block` | 2×2 | Kích thước block |
        | `block_norm` | L2-Hys | Chuẩn hóa block |
        | **Kích thước vector** | **3780** | Chiều đặc trưng |

        #### Tại sao HOG phù hợp với xe?
        - HOG mô tả **hình dạng cục bộ** qua hướng cạnh → bất biến màu sắc
        - Xe có **cấu trúc hình học rõ ràng**: bánh, kính, thân
        - Chuẩn hóa block giúp **bất biến ánh sáng**
        - Nhanh hơn CNN trong môi trường tài nguyên hạn chế
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: HƯỚNG DẪN
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## 🚗 Vehicle Counter — HOG + SVM

    ### Kiến trúc hệ thống

    ```
    Video / Camera
          ↓
    ┌─────────────────────────────────────────────────┐
    │  TIỀN XỬ LÝ                                     │
    │  BGR → Resize(64×128) → Grayscale → CLAHE       │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  TRÍCH XUẤT ĐẶC TRƯNG HOG                       │
    │  9 orientations | 8×8 cell | 2×2 block          │
    │  → Vector 3780 chiều                            │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  IMAGE PYRAMID + SLIDING WINDOW                 │
    │  Scale 1.0 → 1.25 → 1.56 → ...                 │
    │  Step = 8px                                     │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  SVM CLASSIFIER (LinearSVC + Calibration)       │
    │  StandardScaler → predict_proba                 │
    │  conf_thresh = 0.55                             │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  NON-MAXIMUM SUPPRESSION (NMS)                  │
    │  Greedy IoU-based | threshold = 0.3             │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  CENTROID TRACKER                               │
    │  Euclidean matching | max_dist=80px             │
    │  ID assignment | trajectory recording           │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  COUNTING LINE                                   │
    │  Phát hiện xe đi qua đường → Đếm IN/OUT         │
    └─────────────────────────────────────────────────┘
    ```

    ### Cài đặt & Chạy

    ```bash
    # 1. Cài dependencies
    pip install -r requirements.txt

    # 2. Tạo dữ liệu mẫu (nếu không có Kaggle)
    python src/generate_demo_data.py --n-pos 500 --n-neg 500 --video

    # 3. Huấn luyện model
    python src/train.py --pos data/positive --neg data/negative

    # 4. Chạy ứng dụng
    streamlit run app.py
    ```

    ### Tải dataset thật từ Kaggle

    ```bash
    pip install kaggle
    # Đặt file kaggle.json vào ~/.kaggle/
    kaggle datasets download -d sshikamaru/car-object-detection
    # Hoặc: kaggle datasets download -d brsdincer/vehicle-detection-image-set
    unzip *.zip -d data_kaggle/
    # Tách positive/negative theo nhãn trong annotation file
    ```

    ### Cấu trúc thư mục

    ```
    vehicle_counter/
    ├── app.py                  ← Ứng dụng Streamlit
    ├── requirements.txt
    ├── src/
    │   ├── preprocess.py       ← Tiền xử lý ảnh
    │   ├── hog_features.py     ← HOG extraction
    │   ├── train.py            ← Huấn luyện SVM
    │   ├── detector.py         ← Sliding Window + NMS
    │   ├── tracker.py          ← Centroid Tracker + Counting Line
    │   └── generate_demo_data.py ← Tạo dữ liệu mẫu
    ├── models/                 ← Sau khi train
    │   ├── svm_vehicle.pkl
    │   ├── scaler.pkl
    │   └── train_meta.pkl
    └── data/
        ├── positive/           ← Ảnh có xe (64×128)
        ├── negative/           ← Ảnh không xe (64×128)
        └── demo_video.mp4      ← Video test
    ```
    """)
