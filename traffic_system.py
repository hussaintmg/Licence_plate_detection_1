import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import os
import time
from collections import defaultdict
from datetime import datetime

# ==========================================================
#  CONFIG & PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'detections_data')

# Model weights
VEHICLE_MODEL = os.path.join(MODELS_DIR, 'vehicle_detection_model.pkl')
PLATE_MODEL   = os.path.join(MODELS_DIR, 'license_plate_detector.pt')

# Input/Output
# Adjusting to match the project structure found
VIDEO_INPUT   = os.path.join(VIDEOS_DIR, 'output_detection.mp4')
VIDEO_OUTPUT  = os.path.join(VIDEOS_DIR, 'final_traffic_analysis.mp4')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# ==========================================================
#  MODEL LOADING
# ==========================================================
print("🚀 [INIT] Loading YOLO models & EasyOCR...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 [INFO] Running on: {device}")

# Load Vehicle Detection Model
vehicle_net = YOLO(VEHICLE_MODEL)

# Load License Plate Detector
if not os.path.exists(PLATE_MODEL):
    print(f"⚠️ [WARN] License plate model not found at {PLATE_MODEL}. Please ensure it exists.")
plate_net = YOLO(PLATE_MODEL)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
print("✅ [SUCCESS] All systems initialized!")

# ==========================================================
#  UTILITIES
# ==========================================================

def get_plate_text(image):
    """Detects and returns the best OCR text from a cropped plate image."""
    try:
        results = reader.readtext(image)
        if not results:
            return None, 0
        
        # Sort by confidence
        results.sort(key=lambda x: x[2], reverse=True)
        text = results[0][1]
        conf = results[0][2]
        
        # Basic cleanup: uppercase and alphanumeric only
        text = "".join([c for c in text if c.isalnum() or c == ' ']).strip().upper()
        return text, conf
    except Exception as e:
        return None, 0

# ==========================================================
#  MAIN PIPELINE
# ==========================================================

def run_traffic_system():
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"❌ [ERROR] Could not open video: {VIDEO_INPUT}")
        return

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_wr = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    # Counting Line Position (y-coordinate)
    # We place it near the bottom of the frame
    line_y = int(height * 0.82)
    
    # State Management
    # track_plates stores the best OCR result seen for each vehicle ID
    track_plates = {} # track_id -> {'text': str, 'conf': float, 'best_crop': img}
    counted_ids  = set()
    frame_num    = 0
    
    # COCO Classes for vehicles: 2:car, 3:motorcycle, 5:bus, 7:truck
    VEHICLE_IDS = [2, 3, 5, 7]
    
    start_time = time.time()
    print(f"📂 [READY] Processing video: {width}x{height} | {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        # Create a copy for visualization to keep the original clean for processing
        display_frame = frame.copy()
        
        # 1. VEHICLE DETECTION & TRACKING
        # persist=True enables the internal tracker (ByteTrack)
        results = vehicle_net.track(frame, persist=True, verbose=False, classes=VEHICLE_IDS)[0]
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids   = results.boxes.id.cpu().numpy().astype(int)
            clss  = results.boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Protect against out-of-bounds crops
                vy1, vy2 = max(0, y1), min(height, y2)
                vx1, vx2 = max(0, x1), min(width, x2)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                
                # 2. LICENSE PLATE DETECTION (Within Vehicle Crop)
                if vehicle_crop.size > 0:
                    plate_results = plate_net(vehicle_crop, verbose=False, conf=0.4)[0]
                    
                    for p_box in plate_results.boxes.xyxy.cpu().numpy():
                        px1, py1, px2, py2 = map(int, p_box)
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        
                        if plate_crop.size > 0:
                            p_text, p_conf = get_plate_text(plate_crop)
                            
                            if p_text:
                                # Improve OCR by keeping the snapshot with highest confidence
                                current_best = track_plates.get(tid, {'conf': 0})
                                if p_conf > current_best['conf']:
                                    track_plates[tid] = {
                                        'text': p_text,
                                        'conf': p_conf,
                                        'crop': vehicle_crop.copy()
                                    }
                                
                                # Visualize Plate on Main Frame
                                cv2.rectangle(display_frame, (x1+px1, y1+py1), (x1+px2, y1+py2), (0, 255, 0), 2)
                                cv2.putText(display_frame, p_text, (x1+px1, y1+py1-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 3. LINE CROSSING & DATA SAVING
                # Check if this ID has just crossed the line
                if tid not in counted_ids and cy > line_y:
                    counted_ids.add(tid)
                    
                    info = track_plates.get(tid, {'text': 'UNKNOWN', 'crop': vehicle_crop})
                    plate_str = info['text']
                    
                    # Save local image record
                    time_idx = datetime.now().strftime("%H%M%S")
                    filename = f"ID{tid}_{plate_str.replace(' ','_')}_{time_idx}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(save_path, info.get('crop', vehicle_crop))
                    
                    print(f"🚩 [EVENT] ID {tid} crossed | Plate: {plate_str} | Saved: {filename}")

                # 4. VISUALIZATION (Vehicle Boxes)
                cur_plate = track_plates.get(tid, {'text': ''})['text']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                tag = f"ID:{tid} {cur_plate}"
                cv2.putText(display_frame, tag, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

        # 5. UI OVERLAYS (Line and Counters)
        # Draw the virtual detecting line
        cv2.line(display_frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(display_frame, "COUNTING LINE", (15, line_y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # HUD / Dashboard
        cv2.rectangle(display_frame, (10, 10), (320, 80), (0, 0, 0), -1)
        cv2.putText(display_frame, f"VEHICLES: {len(counted_ids)}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"FRAME: {frame_num}/{total_frames}", (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out_wr.write(display_frame)

        # Logging
        if frame_num % 50 == 0:
            elapsed = time.time() - start_time
            print(f"📊 [PROCESS] Frame {frame_num}/{total_frames} | Elapsed: {elapsed:.1f}s")

    # Cleanup
    cap.release()
    out_wr.release()
    
    end_time = time.time()
    print(f"\n✨ [COMPLETE] System Run Statistics:")
    print(f"   - Total Vehicles Counted: {len(counted_ids)}")
    print(f"   - Output Video Location : {VIDEO_OUTPUT}")
    print(f"   - Cropped Images Saved  : {OUTPUT_DIR}")
    print(f"   - Total Processing Time : {end_time - start_time:.1f} seconds")

if __name__ == "__main__":
    run_traffic_system()
