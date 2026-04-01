import cv2
import json
import torch
from ultralytics import YOLO
import os
from datetime import datetime
from utils.tracker import PlateTracker

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

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

def main(video_path=0):
    # Load configuration
    config = load_config()
    
    # Initialize YOLOv11 model (default to yolo11n.pt for speed)
    model = YOLO('yolo11n.pt')
    
    # Initialize Tracker
    tracker = PlateTracker(
        max_age=config.get('TRACKER_MAX_AGE', 30),
        iou_threshold=config.get('IOU_THRESHOLD', 0.3),
        distance_threshold_ratio=config.get('DISTANCE_THRESHOLD_RATIO', 0.5)
    )
    
    # Open Video Source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    # Define vehicle classes (COCO indices: car=2, motorcycle=3, bus=5, truck=7)
    vehicle_classes = config.get('VEHICLE_CLASSES', [2, 3, 5, 7])
    conf_threshold = config.get('CONFIDENCE_THRESHOLD', 0.45)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
            
        # 1. Detection (Only on every Nth frame for performance)
        frame_skip = config.get("FRAME_SKIP", 1)
        if frame_count % frame_skip == 0:
            results = model(frame, verbose=False)[0]
            
            detections = []
            cls_ids = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in vehicle_classes and conf >= conf_threshold:
                    # Get coordinates [x1, y1, x2, y2]
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
        
        # 3. Visualization (Conditional)
        if config.get("show_video", False):
            frame_skip = config.get("FRAME_SKIP", 1)
            for okj in tracked_objects:
                if okj.missing >= frame_skip:
                    continue # Skip drawing lost IDs (allowing for frame skip)
                    
                color = (0, 255, 0) # Green for active
                label = f"ID: {okj.id}"
                
                x1, y1, x2, y2 = map(int, okj.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Vehicle Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Headless mode: no imshow, but we still need a small sleep to avoid 100% CPU lock
            # Or just let it run full speed for maximum performance.
            # Printing status occasionally to know it's working.
            if frame_count % 500 == 0:
                print(f"Processed {frame_count} frames... Active tracks: {len([t for t in tracked_objects if t.missing==0])}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    # Load config first to get RTSP_URL if not provided via CLI
    with open('config.json', 'r') as f:
        conf_data = json.load(f)
    
    video_source = sys.argv[1] if len(sys.argv) > 1 else conf_data.get('RTSP_URL', 0)
    main(video_source)
