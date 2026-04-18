import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import time
import easyocr
import pickle
import re
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from pymongo import MongoClient

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart City — Vehicle Tracker",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stProgress .st-bo { background-color: #00ffe0; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #00ffe0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 2.2rem; font-weight: bold; color: #00ffe0; }
    .metric-lbl { font-size: 0.85rem; color: #aaaaaa; margin-top: 4px; }
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  MONGODB CONNECTION
# ═══════════════════════════════════════════════════════════════════
MONGO_URI = os.getenv("MONGO_URI")

@st.cache_resource
def get_mongo():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db  = client["smart_city"]
        col = db["vehicle_logs"]
        return col
    except Exception as e:
        st.warning(f"⚠️ MongoDB connect nahi hua: {e} — Data sirf screen pe dikhega")
        return None

def save_to_mongo(col, record):
    if col is not None:
        try:
            col.insert_one(record)
        except Exception:
            pass

def fetch_logs(col):
    if col is None:
        return []
    try:
        return list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(500))
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════
#  LOAD MODELS  (cached — sirf ek baar load honge)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    # Vehicle detection model
    vehicle_model = None
    pkl_path = "models/vehicle_detection_model.pkl"
    pt_veh   = "models/vehicle_detection_model.pt"
    if os.path.exists(pkl_path):
        try:
            vehicle_model = YOLO(pkl_path)
        except Exception:
            vehicle_model = YOLO("yolov8n.pt")   # fallback
    elif os.path.exists(pt_veh):
        vehicle_model = YOLO(pt_veh)
    else:
        vehicle_model = YOLO("yolov8n.pt")

    # License plate model
    plate_model = None
    pt_plate = "models/license_plate_detector.pt"
    if os.path.exists(pt_plate):
        plate_model = YOLO(pt_plate)

    # EasyOCR reader
    ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    return vehicle_model, plate_model, ocr_reader

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
CLASS_COLORS    = {
    'car'       : (0,   255,   0),
    'motorcycle': (255, 165,   0),
    'bus'       : (0,   0,   255),
    'truck'     : (0,   255, 255),
}
CLASSES_ORDER = ['car', 'motorcycle', 'bus', 'truck']

# ═══════════════════════════════════════════════════════════════════
#  MOTION ANALYZER  (tumhara notebook wala — same)
# ═══════════════════════════════════════════════════════════════════
def analyze_video_motion(video_path, sample_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    flow_x_list, flow_y_list = [], []
    prev_gray = None
    step = max(1, total // sample_frames)

    for i in range(0, min(total - 1, sample_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = mag > 1.5
            if motion_mask.sum() > 200:
                flow_x_list.append(flow[..., 0][motion_mask].mean())
                flow_y_list.append(flow[..., 1][motion_mask].mean())
        prev_gray = gray

    cap.release()

    if not flow_x_list:
        return {'primary_direction': 'vertical', 'recommended_mode': 'HORIZONTAL',
                'flow_x': 0, 'flow_y': 0, 'angle_deg': 90,
                'width': width, 'height': height, 'fps': fps, 'total': total}

    avg_fx  = np.mean(flow_x_list)
    avg_fy  = np.mean(flow_y_list)
    h_ratio = abs(avg_fx) / (abs(avg_fx) + abs(avg_fy) + 1e-6)
    angle_deg = math.degrees(math.atan2(abs(avg_fy), abs(avg_fx)))

    if h_ratio > 0.65:
        primary, rec_mode = 'horizontal', 'VERTICAL'
        direction = '→ RIGHT' if avg_fx > 0 else '← LEFT'
    elif h_ratio < 0.35:
        primary, rec_mode = 'vertical', 'HORIZONTAL'
        direction = '↓ DOWN' if avg_fy > 0 else '↑ UP'
    else:
        primary, rec_mode = 'diagonal', 'HORIZONTAL'
        direction = f'↘ DIAGONAL ({angle_deg:.0f}°)'

    return {
        'primary_direction': primary, 'recommended_mode': rec_mode,
        'flow_x': avg_fx,  'flow_y': avg_fy,
        'h_ratio': h_ratio, 'angle_deg': angle_deg, 'direction': direction,
        'width': width, 'height': height, 'fps': fps, 'total': total
    }

# ═══════════════════════════════════════════════════════════════════
#  LINE BUILDER  (tumhara notebook wala — same)
# ═══════════════════════════════════════════════════════════════════
def build_counting_line(motion_info, ratio=0.90):
    W    = motion_info['width']
    H    = motion_info['height']
    mode = motion_info['recommended_mode']

    if mode == 'VERTICAL':
        x = int(W * ratio)
        return {
            'mode': 'VERTICAL',
            'pt1' : (x, 0), 'pt2': (x, H),
            'coord': x, 'axis': 'x',
            'label': f'COUNTING LINE (x={x})'
        }
    else:
        y = int(H * ratio)
        return {
            'mode': 'HORIZONTAL',
            'pt1' : (0, y), 'pt2': (W, y),
            'coord': y, 'axis': 'y',
            'label': f'COUNTING LINE (y={y})'
        }

# ═══════════════════════════════════════════════════════════════════
#  SIMPLE TRACKER  (tumhara notebook wala — same)
# ═══════════════════════════════════════════════════════════════════
class SimpleTracker:
    def __init__(self, max_disappeared=40, iou_threshold=0.25):
        self.next_id         = 1
        self.tracks          = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold   = iou_threshold

    def _iou(self, A, B):
        xA = max(A[0], B[0]); yA = max(A[1], B[1])
        xB = min(A[2], B[2]); yB = min(A[3], B[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter == 0: return 0.0
        return inter / float((A[2]-A[0])*(A[3]-A[1]) + (B[2]-B[0])*(B[3]-B[1]) - inter)

    def _register(self, det):
        x1, y1, x2, y2, cls_name, conf = det
        self.tracks[self.next_id] = {
            'bbox'       : (x1, y1, x2, y2),
            'centroid'   : ((x1+x2)//2, (y1+y2)//2),
            'class_name' : cls_name,
            'conf'       : conf,
            'disappeared': 0,
            'counted'    : False,
            'prev_side'  : None
        }
        self.next_id += 1

    def update(self, detections):
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['disappeared'] += 1
                if self.tracks[tid]['disappeared'] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        if not self.tracks:
            for det in detections:
                self._register(det)
        else:
            track_ids   = list(self.tracks.keys())
            track_boxes = [self.tracks[t]['bbox'] for t in track_ids]

            iou_mat = np.zeros((len(track_ids), len(detections)))
            for t, tb in enumerate(track_boxes):
                for d, det in enumerate(detections):
                    iou_mat[t, d] = self._iou(tb, (det[0], det[1], det[2], det[3]))

            matched_t, matched_d = set(), set()
            for idx in np.argsort(iou_mat.ravel())[::-1]:
                t_i, d_i = divmod(idx, len(detections))
                if iou_mat[t_i, d_i] < self.iou_threshold: break
                if t_i in matched_t or d_i in matched_d: continue
                tid = track_ids[t_i]
                x1, y1, x2, y2 = detections[d_i][:4]
                self.tracks[tid].update({
                    'bbox'       : (x1, y1, x2, y2),
                    'centroid'   : ((x1+x2)//2, (y1+y2)//2),
                    'class_name' : detections[d_i][4],
                    'conf'       : detections[d_i][5],
                    'disappeared': 0
                })
                matched_t.add(t_i); matched_d.add(d_i)

            for t_i, tid in enumerate(track_ids):
                if t_i not in matched_t:
                    self.tracks[tid]['disappeared'] += 1
                    if self.tracks[tid]['disappeared'] > self.max_disappeared:
                        del self.tracks[tid]

            for d_i, det in enumerate(detections):
                if d_i not in matched_d:
                    self._register(det)

        return [
            (tid, *info['bbox'], *info['centroid'], info['class_name'], info['conf'])
            for tid, info in self.tracks.items()
        ]

# ═══════════════════════════════════════════════════════════════════
#  CROSSING CHECKER  (tumhara notebook wala — same)
# ═══════════════════════════════════════════════════════════════════
def check_crossing(track_info, line_cfg, tolerance=8):
    if track_info['counted']:
        return False
    cx, cy = track_info['centroid']
    val    = cy if line_cfg['axis'] == 'y' else cx
    coord  = line_cfg['coord']

    current_side = 1 if val > coord else -1
    prev_side    = track_info['prev_side']

    if prev_side is None:
        track_info['prev_side'] = current_side
        return False

    track_info['prev_side'] = current_side
    return abs(val - coord) <= tolerance or (prev_side != current_side)

# ═══════════════════════════════════════════════════════════════════
#  OCR — LICENSE PLATE TEXT EXTRACT
# ═══════════════════════════════════════════════════════════════════
def extract_plate_text(ocr_reader, plate_crop):
    try:
        results = ocr_reader.readtext(plate_crop, detail=0, paragraph=True)
        text    = " ".join(results).strip().upper()
        text    = re.sub(r'[^A-Z0-9\-]', '', text)
        return text if len(text) >= 2 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"

# ═══════════════════════════════════════════════════════════════════
#  DRAW TOP STATS BOX  (video frame pe)
# ═══════════════════════════════════════════════════════════════════
def draw_stats_box(frame, count):
    total        = sum(count.values())
    box_x, box_y = 10, 10
    box_w        = 270
    header_h     = 42
    row_h        = 28
    box_h        = header_h + len(CLASSES_ORDER) * row_h + 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x+box_w, box_y+box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (box_x, box_y),
                  (box_x+box_w, box_y+box_h), (0, 255, 255), 2)

    # TOTAL header
    cv2.putText(frame, f"TOTAL VEHICLES : {total}",
                (box_x+10, box_y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
    cv2.line(frame,
             (box_x+6,       box_y+header_h),
             (box_x+box_w-6, box_y+header_h), (70, 70, 70), 1)

    # Per-class rows
    for i, cls in enumerate(CLASSES_ORDER):
        row_y = box_y + header_h + i * row_h + 22
        color = CLASS_COLORS[cls]
        cv2.circle(frame, (box_x+16, row_y-6), 6, color, -1)
        cv2.putText(frame, f"{cls:<12}: {count[cls]}",
                    (box_x+30, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

# ═══════════════════════════════════════════════════════════════════
#  MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════════
def process_video(video_path, output_path, vehicle_model, plate_model,
                  ocr_reader, mongo_col, line_ratio,
                  progress_bar, status_text, frame_placeholder):

    # ── Step 1: Motion Analysis ──────────────────────────────────────
    status_text.info("🔍 Step 1: Motion analysis chal rahi hai...")
    motion_info = analyze_video_motion(video_path, sample_frames=60)
    if motion_info is None:
        st.error("❌ Video open nahi hua!")
        return None, []

    direction_str = motion_info.get('direction', 'Unknown')
    status_text.info(f"✅ Motion detected: {direction_str}")

    # ── Step 2: Build Line ───────────────────────────────────────────
    line_cfg = build_counting_line(motion_info, ratio=line_ratio)

    # ── Step 3: Video Setup ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    W   = motion_info['width']
    H   = motion_info['height']
    FPS = motion_info['fps'] or 30
    TOT = motion_info['total']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_wr = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    tracker     = SimpleTracker(max_disappeared=40, iou_threshold=0.25)
    count       = defaultdict(int)
    counted_ids = set()
    all_logs    = []
    frame_num   = 0

    status_text.info("🚗 Step 3: Tracking + Detection shuru...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        current_second = round(frame_num / FPS, 2)

        # ── Vehicle Detection ────────────────────────────────────────
        v_results  = vehicle_model(frame, verbose=False)[0]
        detections = []
        for box in v_results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES and conf > 0.30:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls_id], conf))

        # ── Tracker Update ───────────────────────────────────────────
        tracked = tracker.update(detections)

        # ── Draw Counting Line ───────────────────────────────────────
        cv2.line(frame, line_cfg['pt1'], line_cfg['pt2'], (0, 255, 255), 3)
        cv2.putText(frame, line_cfg['label'],
                    (line_cfg['pt1'][0]+8, line_cfg['pt1'][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # ── Process Each Tracked Vehicle ─────────────────────────────
        for (tid, x1, y1, x2, y2, cx, cy, cls_name, conf) in tracked:
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))

            # Vehicle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vehicle name label (top of box)
            v_label = f"{cls_name.upper()}  ID:{tid}"
            (lw, lh), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw+6, y1), color, -1)
            cv2.putText(frame, v_label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

            # Centroid dot
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 8, color, 2)

            # ── License Plate Detection ──────────────────────────────
            plate_text = "UNKNOWN"
            if plate_model is not None:
                vehicle_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if vehicle_crop.size > 0:
                    p_results = plate_model(vehicle_crop, verbose=False)[0]
                    for pbox in p_results.boxes:
                        pconf = float(pbox.conf[0])
                        if pconf > 0.30:
                            px1,py1,px2,py2 = map(int, pbox.xyxy[0])
                            # Plate bounding box (absolute coords)
                            abs_px1 = x1 + px1; abs_py1 = y1 + py1
                            abs_px2 = x1 + px2; abs_py2 = y1 + py2

                            cv2.rectangle(frame,
                                          (abs_px1, abs_py1),
                                          (abs_px2, abs_py2),
                                          (0, 165, 255), 2)

                            # OCR on plate crop
                            plate_crop = vehicle_crop[max(0,py1):py2, max(0,px1):px2]
                            if plate_crop.size > 0:
                                plate_text = extract_plate_text(ocr_reader, plate_crop)

                            # Plate label above plate box
                            p_label = f"PLATE: {plate_text}"
                            (pw, ph), _ = cv2.getTextSize(
                                p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                            cv2.rectangle(frame,
                                          (abs_px1, abs_py1-ph-10),
                                          (abs_px1+pw+6, abs_py1),
                                          (0, 100, 200), -1)
                            cv2.putText(frame, p_label,
                                        (abs_px1+3, abs_py1-4),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.50, (255, 255, 255), 2)
                            break   # sirf pehli/best plate

            # ── Line Crossing Check ──────────────────────────────────
            if tid not in counted_ids:
                track_info = tracker.tracks.get(tid)
                if track_info and check_crossing(track_info, line_cfg, tolerance=8):
                    counted_ids.add(tid)
                    count[cls_name] += 1
                    tracker.tracks[tid]['counted'] = True

                    # Flash ring
                    cv2.circle(frame, (cx, cy), 24, (0, 255, 0), 3)
                    cv2.circle(frame, (cx, cy), 32, (0, 255, 0), 1)

                    # Build log record
                    record = {
                        "track_id"    : int(tid),
                        "vehicle_type": cls_name,
                        "plate_number": plate_text,
                        "frame_number": int(frame_num),
                        "second"      : float(current_second),
                        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence"  : round(float(conf), 2),
                    }
                    all_logs.append(record)
                    save_to_mongo(mongo_col, record.copy())

        # ── Top Stats Box ────────────────────────────────────────────
        draw_stats_box(frame, count)

        # ── Write frame ──────────────────────────────────────────────
        out_wr.write(frame)

        # ── Streamlit live preview (har 15 frames) ───────────────────
        if frame_num % 15 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            progress_bar.progress(min(frame_num / max(TOT, 1), 1.0))
            status_text.info(
                f"⚙️ Frame {frame_num}/{TOT} | "
                f"Counted: {sum(count.values())} | "
                f"Time: {current_second:.1f}s"
            )

    cap.release()
    out_wr.release()
    return count, all_logs

# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════
st.title("🚦 Smart City — Vehicle Tracking & License Plate System")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    line_ratio  = st.slider("Counting Line Position", 0.50, 0.95, 0.90, 0.05,
                             help="Frame ki kitni % height pe line ho (default 90%)")
    conf_thresh = st.slider("Detection Confidence", 0.20, 0.80, 0.30, 0.05)
    st.markdown("---")
    st.markdown("**Models Path:**")
    st.code("models/vehicle_detection_model.pkl\nmodels/license_plate_detector.pt")
    st.markdown("**MongoDB:** `localhost:27017`")
    st.markdown("**DB:** `smart_city`  |  **Collection:** `vehicle_logs`")

# ── Load models ───────────────────────────────────────────────────
with st.spinner("🔧 Models load ho rahe hain..."):
    vehicle_model, plate_model, ocr_reader = load_models()

col1, col2, col3 = st.columns(3)
with col1:
    st.success("✅ Vehicle Model Ready")
with col2:
    st.success("✅ Plate Model Ready") if plate_model else st.warning("⚠️ Plate Model Missing")
with col3:
    st.success("✅ OCR Ready")

st.markdown("---")

# ── MongoDB ───────────────────────────────────────────────────────
mongo_col = get_mongo()

# ── File Upload ───────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Traffic Video Upload Karo",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    input_path  = tfile.name
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    st.video(input_path)

    if st.button("🚀 Tracking Shuru Karo", type="primary", use_container_width=True):
        st.markdown("---")
        st.subheader("📡 Live Processing")

        progress_bar      = st.progress(0)
        status_text       = st.empty()
        frame_placeholder = st.empty()

        # ── Run Processing ────────────────────────────────────────────
        final_count, all_logs = process_video(
            video_path    = input_path,
            output_path   = output_path,
            vehicle_model = vehicle_model,
            plate_model   = plate_model,
            ocr_reader    = ocr_reader,
            mongo_col     = mongo_col,
            line_ratio    = line_ratio,
            progress_bar  = progress_bar,
            status_text   = status_text,
            frame_placeholder = frame_placeholder
        )

        progress_bar.progress(1.0)
        status_text.success("✅ Processing Complete!")
        frame_placeholder.empty()

        # ═══════════════════════════════════════════════════════════
        #  RESULTS
        # ═══════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("📊 Final Results")

        # Metric cards — top row
        total = sum(final_count.values()) if final_count else 0
        m_cols = st.columns(5)
        metrics = [("🚗 Total", total, "#00ffe0"),
                   ("🚙 Cars",  final_count.get('car', 0),        "#00ff00"),
                   ("🛵 Bikes", final_count.get('motorcycle', 0), "#ffa500"),
                   ("🚌 Buses", final_count.get('bus', 0),        "#0055ff"),
                   ("🚛 Trucks",final_count.get('truck', 0),      "#00ffff")]
        for col, (lbl, val, clr) in zip(m_cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:{clr}">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Output Video Download ────────────────────────────────────
        if os.path.exists(output_path):
            with open(output_path, "rb") as vf:
                st.download_button(
                    label     = "⬇️ Tracked Video Download Karo",
                    data      = vf,
                    file_name = "tracked_output.mp4",
                    mime      = "video/mp4",
                    use_container_width=True
                )

        # ── Logs Table ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Vehicle Crossing Log (MongoDB + Local)")

        if all_logs:
            import pandas as pd

            df = pd.DataFrame(all_logs)
            df = df[["track_id", "vehicle_type", "plate_number",
                     "second", "frame_number", "confidence", "timestamp"]]
            df.columns = ["Track ID", "Vehicle Type", "Plate Number",
                          "Second", "Frame", "Confidence", "Timestamp"]
            df = df.sort_values("Second").reset_index(drop=True)

            # Filter controls
            fc1, fc2 = st.columns(2)
            with fc1:
                vtype_filter = st.multiselect(
                    "Vehicle Type Filter",
                    options=CLASSES_ORDER,
                    default=CLASSES_ORDER
                )
            with fc2:
                plate_filter = st.text_input("Plate Number Search", "")

            filtered = df[df["Vehicle Type"].isin(vtype_filter)]
            if plate_filter:
                filtered = filtered[
                    filtered["Plate Number"].str.contains(
                        plate_filter.upper(), na=False)
                ]

            st.dataframe(
                filtered,
                use_container_width=True,
                height=400
            )

            st.caption(f"Total records: {len(filtered)}")

            # CSV download
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ CSV Download Karo",
                data      = csv,
                file_name = "vehicle_logs.csv",
                mime      = "text/csv"
            )
        else:
            st.info("Koi vehicle line cross nahi ki abhi tak.")

# ═══════════════════════════════════════════════════════════════════
#  BOTTOM — MongoDB Previous Logs
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("🗄️ MongoDB — Pichle Saare Logs Dekho"):
    if mongo_col is not None:
        prev_logs = fetch_logs(mongo_col)
        if prev_logs:
            import pandas as pd
            pdf = pd.DataFrame(prev_logs)
            st.dataframe(pdf, use_container_width=True, height=350)
            st.caption(f"Total DB records: {len(prev_logs)}")
        else:
            st.info("Database mein abhi koi record nahi hai.")
    else:
        st.warning("MongoDB connected nahi hai.")