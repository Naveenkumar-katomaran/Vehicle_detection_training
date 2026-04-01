import cv2
import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from utils.tracker import PlateTracker

def load_config(config_path='config.json'):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {}

def save_training_data(frame, visible_objects, vehicle_classes, frame_count):
    """Saves frame and labels all currently visible objects."""
    if not visible_objects:
        return
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_dir = os.path.join("training", date_str)
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # Save classes.txt if not exists
    class_map = {cls_id: i for i, cls_id in enumerate(vehicle_classes)}
    class_names = ["car", "motorcycle", "bus", "truck"]
    classes_path = os.path.join(base_dir, "classes.txt")
    if not os.path.exists(classes_path):
        with open(classes_path, "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%H%M%S_%f")
    filename = f"train_{timestamp}"
    
    # Save Image
    img_path = os.path.join(img_dir, f"{filename}.jpg")
    cv2.imwrite(img_path, frame)
    
    # Save Labels
    h, w = frame.shape[:2]
    lbl_path = os.path.join(lbl_dir, f"{filename}.txt")
    with open(lbl_path, "w") as f:
        for obj in visible_objects:
            if obj.cls_id is None:
                continue
            x1, y1, x2, y2 = obj.bbox
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            bx = (x1 + x2) / (2 * w)
            by = (y1 + y2) / (2 * h)
            
            mapped_cls = class_map.get(obj.cls_id, 0)
            f.write(f"{mapped_cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
            obj.saved_count += 1
            obj.last_saved_frame = frame_count

def draw_premium_bbox(frame, bbox, obj_id, is_lost=False, fps=None):
    """Draws a premium-looking bounding box with ID and optional FPS."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Colors (vibrant and professional)
    # Green for active, Yellow/Orange for lost/ghosting
    color = (0, 255, 127) if not is_lost else (0, 165, 255)
    thickness = 2
    
    # 1. Draw Bounding Box with rounded-like corners (simulated by lines)
    length = 20
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness + 1)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness + 1)
    
    # Thin box for the rest
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # 2. Draw Label Background
    label = f"ID: {obj_id}"
    if is_lost:
        label += " (Lost)"
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    (w, h), _ = cv2.getTextSize(label, font, font_scale, 1)
    
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 7), font, font_scale, (255, 255, 255), 1)

def main():
    parser = argparse.ArgumentParser(description="Premium Vehicle Detection and Tracking")
    parser.add_argument("--source", type=str, default="0", help="Video source (file path or camera index)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--show", action="store_true", default=True, help="Display the output window")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Weights and Thresholds
    model = YOLO(args.model)
    vehicle_classes = config.get('VEHICLE_CLASSES', [2, 3, 5, 7]) # car, motorcycle, bus, truck
    conf_threshold = config.get('CONFIDENCE_THRESHOLD', 0.45)
    
    # Initialize Tracker
    tracker = PlateTracker(
        max_age=config.get('TRACKER_MAX_AGE', 30),
        iou_threshold=config.get('IOU_THRESHOLD', 0.3),
        distance_threshold_ratio=config.get('DISTANCE_THRESHOLD_RATIO', 0.5)
    )

    # Capture Video
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    print(f"Starting tracking on {source}...")
    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 1. Run YOLO Detection (Only on every Nth frame for performance)
        frame_skip = config.get("FRAME_SKIP", 1)
        if frame_count % frame_skip == 0:
            results = model(frame, verbose=False)[0]
            
            detections = []
            cls_ids = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in vehicle_classes and conf >= conf_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    detections.append(xyxy)
                    cls_ids.append(cls)
            
            # 2. Update Tracker with detections
            tracked_objects = tracker.update(detections, cls_ids)

            # 2.1 Save Training Data (Triggered if any vehicle needs a sample and interval passed)
            if config.get("Training", False):
                sample_limit = config.get("TRAINING_SAMPLES_PER_VEHICLE", 3)
                interval = config.get("TRAINING_FRAME_INTERVAL", 30)
                visible_objects = [obj for obj in tracked_objects if obj.missing == 0]
                
                # Check if at least one visible object still needs samples AND interval has passed for it
                needs_saving = any(
                    obj.saved_count < sample_limit and (frame_count - obj.last_saved_frame) >= interval 
                    for obj in visible_objects
                )
                
                if needs_saving:
                    save_training_data(frame, visible_objects, vehicle_classes, frame_count)
        else:
            # On skipped frames, update tracker with NO detections to maintain velocity-based prediction
            tracked_objects = tracker.update([], [])
        
        # 3. Premium Visualization (Conditional)
        show_video = args.show or config.get("show_video", False)
        if show_video:
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicles: {len([t for t in tracked_objects if t.missing == 0])}", 
                        (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

            for obj in tracked_objects:
                if obj.missing < frame_skip:
                    draw_premium_bbox(frame, obj.bbox, obj.id, is_lost=False)

            cv2.imshow("Premium Vehicle Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames... Active tracks: {len([t for t in tracked_objects if t.missing==0])}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")

if __name__ == "__main__":
    main()
