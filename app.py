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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .main { 
        background-color: #0b0e14;
        background-image: radial-gradient(circle at 50% 0%, #1c2331 0%, #0b0e14 100%);
    }
    
    .stApp { background: transparent; }
    
    /* Premium Metric Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        flex: 1;
        min-width: 180px;
        background: rgba(30, 36, 50, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 224, 0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: left;
        transition: transform 0.3s ease, border-color 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 255, 224, 0.4);
    }
    
    .metric-val { 
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #00ffe0;
        text-shadow: 0 0 10px rgba(0, 255, 224, 0.3);
    }
    
    .metric-lbl { 
        font-size: 0.9rem; 
        color: #9da5b4; 
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Progress and Status */
    .stProgress .st-bo { background: linear-gradient(90deg, #00ffe0, #0088ff); }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #9da5b4;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00ffe0 !important;
        border-bottom: 2px solid #00ffe0 !important;
    }

    /* DataFrame styling */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .status-msg {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(0, 255, 224, 0.05);
        border: 1px solid rgba(0, 255, 224, 0.2);
        color: #00ffe0;
        margin-bottom: 1rem;
    }
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
        st.warning(f"⚠️ MongoDB connection failed: {e} — Data will only be displayed on screen.")
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

    if os.path.exists(pt_veh):
        try:
            vehicle_model = YOLO(pt_veh)
        except Exception:
            vehicle_model = YOLO("yolov8n.pt")
    elif os.path.exists(pkl_path):
        # If it's a .pkl, it might be a custom non-YOLO model. 
        # YOLO() might fail on it, so we fallback to a standard model for compatibility.
        vehicle_model = YOLO("yolov8n.pt") 
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
#  OCR — LICENSE PLATE TEXT EXTRACT (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════
def extract_plate_text(ocr_reader, plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return "UNKNOWN"
    try:
        # Preprocessing for better OCR
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Resize to at least 320px width if small
        h, w = gray.shape
        if w < 320:
            scale = 320 / w
            gray  = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            
        # Optional: Contrast enhancement
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
        
        results = ocr_reader.readtext(gray, detail=0, paragraph=True)
        text    = " ".join(results).strip().upper()
        # Filter for alphanumeric characters
        text    = re.sub(r'[^A-Z0-9]', '', text)
        
        # Validation: Most license plates have 4-10 chars
        if 3 <= len(text) <= 12:
            return text
        return "UNKNOWN"
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
#  MAIN PROCESSING FUNCTION (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════
def process_video(video_path, output_path, vehicle_model, plate_model,
                  ocr_reader, mongo_col, line_ratio,
                  progress_bar, status_text, frame_placeholder, show_preview=True):

    # ── Step 1: Motion Analysis ──────────────────────────────────────
    status_text.markdown('<div class="status-msg">🔍 Step 1: Analyzing motion patterns...</div>', unsafe_allow_html=True)
    motion_info = analyze_video_motion(video_path, sample_frames=60)
    if motion_info is None:
        st.error("❌ Could not open video file!")
        return None, []

    direction_str = motion_info.get('direction', 'Unknown')
    status_text.markdown(f'<div class="status-msg">✅ Motion detected: <b>{direction_str}</b></div>', unsafe_allow_html=True)

    # ── Step 2: Build Line ───────────────────────────────────────────
    line_cfg = build_counting_line(motion_info, ratio=line_ratio)

    # ── Step 3: Video Setup ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    W   = motion_info['width']
    H   = motion_info['height']
    FPS = motion_info['fps'] or 30
    TOT = motion_info['total']

    # Using 'avc1' for H.264
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_wr = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))
    
    if not out_wr.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_wr = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    tracker      = SimpleTracker(max_disappeared=40, iou_threshold=0.25)
    count        = defaultdict(int)
    counted_ids  = set()
    best_plates  = {}  # Cache: {track_id: "BEST_TEXT"}
    all_logs     = []
    frame_num    = 0

    status_text.markdown('<div class="status-msg">🚗 Step 2: Processing frames & detecting vehicles...</div>', unsafe_allow_html=True)

    # ROI for Plate Detection: Only process near the counting line (±15% of frame)
    roi_threshold = 0.15 * (H if line_cfg['axis'] == 'y' else W)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        current_second = round(frame_num / FPS, 2)

        # ── Vehicle Detection ────────────────────────────────────────
        v_results  = vehicle_model(frame, verbose=False, stream=True)
        detections = []
        for result in v_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id in VEHICLE_CLASSES and conf > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls_id], conf))

        # ── Tracker Update ───────────────────────────────────────────
        tracked = tracker.update(detections)

        # ── Draw Counting Line ───────────────────────────────────────
        cv2.line(frame, line_cfg['pt1'], line_cfg['pt2'], (0, 255, 255), 3)

        # ── Process Each Tracked Vehicle ─────────────────────────────
        for (tid, x1, y1, x2, y2, cx, cy, cls_name, conf) in tracked:
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Get plate text from cache or detect
            plate_text = best_plates.get(tid, "UNKNOWN")
            
            # ── Optimized License Plate Detection ────────────────────
            # Trigger ONLY if: 
            # 1. We don't have a good plate yet OR current is "UNKNOWN"
            # 2. Vehicle is near the counting line (ROI)
            val    = cy if line_cfg['axis'] == 'y' else cx
            dist   = abs(val - line_cfg['coord'])
            
            if (plate_text == "UNKNOWN") and (dist < roi_threshold) and (plate_model is not None):
                vehicle_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if vehicle_crop.size > 0:
                    p_results = plate_model(vehicle_crop, verbose=False)[0]
                    for pbox in p_results.boxes:
                        if float(pbox.conf[0]) > 0.30:
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                            
                            abs_px1, abs_py1 = x1 + px1, y1 + py1
                            abs_px2, abs_py2 = x1 + px2, y1 + py2
                            cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 165, 255), 2)

                            plate_crop = vehicle_crop[max(0,py1):py2, max(0,px1):px2]
                            found_text = extract_plate_text(ocr_reader, plate_crop)
                            
                            if found_text != "UNKNOWN":
                                best_plates[tid] = found_text
                                plate_text = found_text
                            break

            # ── Dynamic Label ────────────────────────────────────────
            display_label = f"{cls_name.upper()} #{tid} | {plate_text}"
            (lw, lh), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            cv2.rectangle(frame, (x1, y1-lh-15), (x1+lw+10, y1), color, -1)
            cv2.putText(frame, display_label, (x1+5, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)

            # ── Line Crossing Check ──────────────────────────────────
            if tid not in counted_ids:
                track_info = tracker.tracks.get(tid)
                if track_info and check_crossing(track_info, line_cfg, tolerance=12):
                    counted_ids.add(tid)
                    count[cls_name] += 1
                    tracker.tracks[tid]['counted'] = True

                    # Event log
                    record = {
                        "track_id": int(tid),
                        "vehicle_type": cls_name,
                        "plate_number": plate_text,
                        "frame_number": int(frame_num),
                        "second": float(current_second),
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "confidence": round(float(conf), 2),
                    }
                    all_logs.append(record)
                    save_to_mongo(mongo_col, record.copy())

        # ── Annotatons ───────────────────────────────────────────────
        draw_stats_box(frame, count)
        out_wr.write(frame)

        # ── UI Update (Throttled even more for Speed) ───────────────
        if frame_num % 45 == 0 or frame_num == TOT:
            progress = min(frame_num / max(TOT, 1), 1.0)
            progress_bar.progress(progress)
            
            if show_preview:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            
            status_text.markdown(f"""
            <div style="color: #00ffe0; font-size: 0.9rem; margin-top: 10px;">
                Processing: <b>{frame_num}/{TOT}</b> | 
                Counted: <b>{sum(count.values())}</b> | 
                Latest Plate: <b>{list(best_plates.values())[-1] if best_plates else 'N/A'}</b>
            </div>
            """, unsafe_allow_html=True)

    cap.release()
    out_wr.release()
    return count, all_logs


# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

# Header with custom styling
st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <span style="font-size: 3rem; margin-right: 1.5rem;">🚦</span>
        <div>
            <h1 style="margin: 0; color: #ffffff; font-weight: 700;">Smart City AI</h1>
            <p style="margin: 0; color: #00ffe0; letter-spacing: 2px; font-weight: 300;">NEXT-GEN TRAFFIC ANALYTICS SYSTEM</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ CONFIGURATION")
    line_ratio  = st.slider("Counting Line Position", 0.50, 0.95, 0.90, 0.05,
                             help="Position of the counting line as % of frame height.")
    conf_thresh = st.slider("Detection Confidence", 0.20, 0.80, 0.30, 0.05)
    
    st.markdown("---")
    st.markdown("### 💡 PREFERENCE")
    show_preview = st.checkbox("Show Real-time Preview", value=True, help="Turning this off speeds up processing.")
    
    st.markdown("---")
    # GitHub Integration Hint
    st.info("📦 **Repository Status**\nOrigin: hussaintmg/Licence_plate_detection_1\nBranch: main")

# ── Initialize session state ──────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None

# ── Main Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 UPLOAD & PROCESS", "📊 ANALYTICS", "🗄️ HISTORY"])

with tab1:
    # ── File Upload ───────────────────────────────────────────────
    uploaded = st.file_uploader(
        "DROP TRAFFIC VIDEO OR CLICK TO UPLOAD",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if uploaded:
        # Save to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        input_path  = tfile.name
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 原 Source Video")
            st.video(input_path)
        
        with c2:
            st.markdown("#### ⚙️ Operations")
            if st.button("🚀 INITIATE ANALYSIS", type="primary", use_container_width=True):
                # UI Placeholders
                prog_container = st.container()
                with prog_container:
                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    frame_placeholder = st.empty()

                # Process
                vehicle_model, plate_model, ocr_reader = load_models()
                mongo_col = get_mongo()
                final_count, all_logs = process_video(
                    video_path = input_path,
                    output_path = output_path,
                    vehicle_model = vehicle_model,
                    plate_model = plate_model,
                    ocr_reader = ocr_reader,
                    mongo_col = mongo_col,
                    line_ratio = line_ratio,
                    progress_bar = progress_bar,
                    status_text = status_text,
                    frame_placeholder = frame_placeholder,
                    show_preview = show_preview
                )

                st.session_state.results = {
                    'count': final_count,
                    'logs': all_logs,
                    'output_path': output_path
                }
                st.success("Analysis Complete!")
                st.rerun()

        # Display Results if available
        if st.session_state.results:
            res = st.session_state.results
            st.markdown("---")
            st.markdown("### 🎥 TRACKED OUTPUT")
            
            out_c1, out_c2 = st.columns([2, 1])
            
            with out_c1:
                if os.path.exists(res['output_path']):
                    # Reading as bytes often fixes dashboard playback issues in Streamlit
                    with open(res['output_path'], "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
            
            with out_c2:

                st.markdown("#### 📥 DOWNLOADS")
                with open(res['output_path'], "rb") as f:
                    st.download_button("💾 DOWNLOAD PROCESSED VIDEO", f, "processed_traffic.mp4", "video/mp4", use_container_width=True)
                
                if res['logs']:
                    import pandas as pd
                    pdf = pd.DataFrame(res['logs'])
                    csv = pdf.to_csv(index=False).encode('utf-8')
                    st.download_button("📊 DOWNLOAD EVENT LOG (CSV)", csv, "traffic_logs.csv", "text/csv", use_container_width=True)

with tab2:
    if st.session_state.results:
        res = st.session_state.results
        counts = res['count']
        
        # Metric Grid
        import pandas as pd
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, cls in enumerate(CLASSES_ORDER):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">{cls}s detected</div>
                    <div class="metric-val">{counts.get(cls, 0)}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Log Table
        if res['logs']:
            st.markdown("### 📋 EVENT TIMELINE")
            df = pd.DataFrame(res['logs'])
            st.dataframe(df, use_container_width=True, height=500)
    else:
        st.info("Analysis required to view analytics. Please upload and process a video.")

with tab3:
    st.markdown("### 💾 CENTRALIZED CLOUD STORAGE (MONGODB)")
    mongo_col = get_mongo()
    if mongo_col is not None:
        prev_logs = fetch_logs(mongo_col)
        if prev_logs:
            import pandas as pd
            pdf = pd.DataFrame(prev_logs)
            st.dataframe(pdf, use_container_width=True, height=600)
        else:
            st.info("No records found in cloud database.")
    else:
        st.warning("MongoDB not connected. Check your MONGO_URI environment variable.")