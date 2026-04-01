import cv2
import json
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import threading
import time
import queue
import logging
from utils.tracker import PlateTracker

# Global dictionary to store latest frames
preview_frames = {}
preview_lock = threading.Lock()
running = True

class CustomFormatter(logging.Formatter):
    """Custom format: [01-Apr-26 17:12:36.693] [INFO] [utils.tracker] [System] - Message"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%d-%b-%y %H:%M:%S.%f")[:-3]

    def format(self, record):
        record.asctime = self.formatTime(record)
        return f"[{record.asctime}] [{record.levelname}] [{record.name}] [System] - {record.getMessage()}"

def setup_logging():
    """Initialize logging into date-wise folders."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("logs", date_str)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "vehicle_detection.log")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    formatter = CustomFormatter()
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    
    logging.info("Logging initialized in date-wise folder.")

def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_training_data(frame, visible_objects, vehicle_classes, frame_count, camera_name="cam"):
    """Saves frame and labels all currently visible objects into camera-specific folders."""
    if not visible_objects:
        return
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_dir = os.path.join("training", date_str, camera_name)
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # Save classes.txt in the camera folder if not exists
    class_map = {cls_id: i for i, cls_id in enumerate(vehicle_classes)}
    class_names = ["car", "motorcycle", "bus", "truck"]
    classes_path = os.path.join(base_dir, "classes.txt")
    if not os.path.exists(classes_path):
        with open(classes_path, "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%H%M%S_%f")
    filename = f"{timestamp}"
    
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

def process_camera(camera_name, video_path, config, model):
    """Process a single camera stream in a thread with individual logging."""
    global running
    
    # Initialize individual logger for this camera
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("logs", date_str)
    os.makedirs(log_dir, exist_ok=True)
    
    cam_logger = logging.getLogger(camera_name)
    cam_logger.setLevel(logging.INFO)
    cam_logger.propagate = False # don't send to root logger to avoid duplication
    
    # Add file handler for this camera
    fh = logging.FileHandler(os.path.join(log_dir, f"{camera_name}.log"))
    fh.setFormatter(CustomFormatter())
    cam_logger.addHandler(fh)
    
    # Add console handler for visibility
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    cam_logger.addHandler(ch)

    cam_logger.info(f"Starting camera thread on {video_path}")
    
    # Initialize Tracker for this specific camera
    tracker = PlateTracker(
        max_age=config.get('TRACKER_MAX_AGE', 5),
        iou_threshold=config.get('IOU_THRESHOLD', 0.5),
        distance_threshold=config.get('DISTANCE_THRESHOLD', 300),
        distance_scale_factor=config.get('DISTANCE_SCALE_FACTOR', 1.5)
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cam_logger.error(f"Error: Could not open camera (URL: {video_path})")
        return

    vehicle_classes = config.get('VEHICLE_CLASSES', [2, 3, 5, 7])
    conf_threshold = config.get('CONFIDENCE_THRESHOLD', 0.45)
    frame_count = 0
    show_video = config.get("show_video", False)
    frame_skip = config.get("FRAME_SKIP", 1)
    
    conf_device = config.get("device", "auto").lower()
    if conf_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif conf_device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    use_half = (device == "cuda")

    failed_frames = 0
    try:
        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret:
                failed_frames += 1
                if failed_frames >= 5:
                    cam_logger.warning(f"Connection lost (5 failed frames). Reconnecting...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(video_path)
                    failed_frames = 0
                continue
                
            failed_frames = 0
            frame_count += 1
                
            # 1. Detection
            if frame_count % frame_skip == 0:
                results = model(frame, verbose=False, half=use_half, device=device)[0]
                
                detections, confs, cls_ids = [], [], []
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in vehicle_classes and conf >= conf_threshold:
                        detections.append(box.xyxy[0].cpu().numpy())
                        confs.append(conf)
                        cls_ids.append(cls)
                
                # 2. Update Tracker
                tracker.update(detections, cls_ids, confs, frame)
                
                # Memory optimization
                del results
                if device == "cuda" and frame_count % 100 == 0:
                    torch.cuda.empty_cache()

                # 2.1 Save Training Data
                if config.get("Training", False):
                    sample_limit = config.get("TRAINING_SAMPLES_PER_VEHICLE", 3)
                    interval = config.get("TRAINING_FRAME_INTERVAL", 30)
                    visible_objects = [obj for obj in tracker.objects if obj.missing_frames == 0]
                    
                    needs_saving = any(
                        obj.saved_count < sample_limit and (frame_count - obj.last_saved_frame) >= interval 
                        for obj in visible_objects
                    )
                    
                    if needs_saving:
                        for obj in visible_objects:
                            obj.bbox = obj.bboxes[-1]
                        save_training_data(frame, visible_objects, vehicle_classes, frame_count, camera_name)
            else:
                tracker.update([], [], [], frame)
            
            # 3. Handle Frames for Preview
            if show_video:
                vis_frame = frame.copy()
                for okj in tracker.objects:
                    if okj.missing_frames >= frame_skip:
                        continue
                    color = (0, 255, 0)
                    x1, y1, x2, y2 = map(int, okj.bboxes[-1])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_frame, f"{camera_name} ID: {okj.obj_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                with preview_lock:
                    preview_frames[camera_name] = vis_frame
            else:
                if frame_count % 500 == 0:
                    cam_logger.info(f"Processed {frame_count} frames... Active tracks: {len([t for t in tracker.objects if t.missing_frames==0])}")
    except Exception as e:
        cam_logger.error(f"Error in camera thread: {e}")
    finally:
        cap.release()
        cam_logger.info("Camera thread shutting down.")

def main():
    global running
    setup_logging()
    config = load_config()
    model = YOLO('yolo11n.pt')
    
    conf_device = config.get("device", "auto").lower()
    if conf_device == "cuda" and torch.cuda.is_available():
        target_device = "cuda"
    elif conf_device == "cpu":
        target_device = "cpu"
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Device configuration: {conf_device} (Effective: {target_device})")
    if target_device == "cuda":
        model.to("cuda")
    
    camera_urls = config.get("camera_url", {})
    enabled_cameras = config.get("enabled_cameras", [])
    
    threads = []
    for cam_name in enabled_cameras:
        url = camera_urls.get(cam_name)
        if url:
            t = threading.Thread(target=process_camera, args=(cam_name, url, config, model), daemon=True)
            t.start()
            threads.append(t)
        else:
            logging.warning(f"Camera '{cam_name}' enabled but no URL found in config.")

    show_video = config.get("show_video", False)
    
    try:
        while running and any(t.is_alive() for t in threads):
            if show_video:
                with preview_lock:
                    current_previews = list(preview_frames.items())
                
                for cam_name, frame in current_previews:
                    cv2.imshow(f"Tracking: {cam_name}", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Shutdown requested...")
        running = False

    logging.info("Cleaning up...")
    running = False
    for t in threads:
        t.join(timeout=2)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
