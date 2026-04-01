import numpy as np
import time
import logging as log
import cv2

logging = log.getLogger(__name__)

def get_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_centroid(box):
    # box format: [x1, y1, x2, y2]
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

class TrackedObject:
    def __init__(self, obj_id, frame, bbox, conf, cls_id, max_batch_size=30):
        self.obj_id = obj_id
        self.cls_id = cls_id
        self.images = [] # Store CROPS only
        self.bboxes = [] # Stores [x1, y1, x2, y2]
        self.confs = []
        self.best_full_frame = None # Single high-quality overview shot
        self.best_conf = -1.0
        self.best_plate_bbox = None # Store the bbox matching the best_full_frame
        
        self.last_seen = time.time()
        self.max_batch_size = max_batch_size
        self.is_validated = False
        self.needs_flush = False
        self.has_ended = False
        self.matched_this_cycle = False
        self.detection_updates = 0
        self.velocity = (0, 0) # (vx, vy) pixels per frame
        self.missing_frames = 0
        self.saved_count = 0 # For training data limiting
        self.last_saved_frame = -100
        
        self.start_centroid = get_centroid(bbox)

        # Add the first detection
        self.add_detection(frame, bbox, conf)

    def predict(self):
        """Predict the bounding box in the next frame based on constant velocity."""
        last_bbox = self.bboxes[-1]
        x1, y1, x2, y2 = last_bbox
        vx, vy = self.velocity
        
        multiplier = self.missing_frames + 1
        return [
            x1 + vx * multiplier, y1 + vy * multiplier,
            x2 + vx * multiplier, y2 + vy * multiplier
        ]

    def add_detection(self, frame, bbox, conf):
        # Update velocity if we have at least one previous box
        if len(self.bboxes) == 1:
            c_prev = get_centroid(self.bboxes[-1])
            c_now = get_centroid(bbox)
            self.velocity = (c_now[0] - c_prev[0], c_now[1] - c_prev[1])
        elif len(self.bboxes) > 1:
            c_prev = get_centroid(self.bboxes[-1])
            c_now = get_centroid(bbox)
            vx_new = c_now[0] - c_prev[0]
            vy_new = c_now[1] - c_prev[1]
            # Smooth velocity (70% new, 30% old)
            self.velocity = (
                0.7 * vx_new + 0.3 * self.velocity[0],
                0.7 * vy_new + 0.3 * self.velocity[1]
            )

        # Update lists
        self.bboxes.append(bbox)
        self.confs.append(conf)

        # CROPPING LOGIC: Store only the vehicle area
        try:
            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            # Expand slightly for robustness
            ex1, ey1 = max(0, x1 - 10), max(0, y1 - 10)
            ex2, ey2 = min(fw, x2 + 10), min(fh, y2 + 10)
            
            if ex2 > ex1 and ey2 > ey1:
                crop = frame[ey1:ey2, ex1:ex2].copy()
                self.images.append(crop)
                
                # Update Best Full Frame (Overview Shot)
                if conf > self.best_conf:
                    self.best_conf = conf
                    self.best_full_frame = frame.copy() 
                    self.best_plate_bbox = bbox
        except Exception as e:
            logging.error(f"[Tracker] Cropping error: {e}")

        self.detection_updates += 1
        if len(self.images) >= self.max_batch_size:
            self.needs_flush = True

        self.last_seen = time.time()
        self.missing_frames = 0

class PlateTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, max_batch_size=30, distance_threshold=300, distance_scale_factor=1.5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_batch_size = max_batch_size
        self.objects = []
        self.next_id = 0
        self.frame_count = 0
        self.distance_threshold = distance_threshold
        self.distance_scale_factor = distance_scale_factor

    def update(self, detections, cls_ids, confidences, frame):
        self.frame_count += 1
        
        active_objects = [o for o in self.objects if not o.has_ended]
        
        # 1. New objects if none active
        if not active_objects:
            for i, det in enumerate(detections):
                new_obj = TrackedObject(self.next_id, frame, det, confidences[i], cls_ids[i], self.max_batch_size)
                self.objects.append(new_obj)
                self.next_id += 1
            return []

        det_centroids = [get_centroid(d) for d in detections]
        costs = []
        
        for obj_idx, obj in enumerate(active_objects):
            predicted_bbox = obj.predict()
            p_centroid = get_centroid(predicted_bbox)
            
            last_bbox = obj.bboxes[-1]
            obj_w = last_bbox[2] - last_bbox[0]
            
            if obj.detection_updates < 2:
                dynamic_dist_limit = self.distance_threshold * 0.8
            else:
                dynamic_dist_limit = obj_w * self.distance_scale_factor
                dynamic_dist_limit = min(dynamic_dist_limit, self.distance_threshold)
            
            dynamic_dist_limit_sq = dynamic_dist_limit ** 2
            
            for det_idx, det in enumerate(detections):
                # Ensure class matching (strictly track same class)
                if cls_ids[det_idx] != obj.cls_id:
                    continue
                    
                iou_val = get_iou(predicted_bbox, det)
                det_centroid = det_centroids[det_idx]
                dist_sq = (p_centroid[0] - det_centroid[0])**2 + (p_centroid[1] - det_centroid[1])**2
                
                if iou_val > self.iou_threshold:
                    cost = 1.0 - iou_val
                    costs.append((cost, obj_idx, det_idx, "IOU"))
                elif dist_sq < dynamic_dist_limit_sq:
                    cost = 1.1 + (dist_sq / dynamic_dist_limit_sq)
                    costs.append((cost, obj_idx, det_idx, "Dist"))

        costs.sort(key=lambda x: x[0])
        matched_objs = set()
        matched_dets = set()
        
        for cost, obj_idx, det_idx, method in costs:
            if obj_idx in matched_objs or det_idx in matched_dets:
                continue
            
            obj = active_objects[obj_idx]
            obj.add_detection(frame, detections[det_idx], confidences[det_idx])
            obj.matched_this_cycle = True
            matched_objs.add(obj_idx)
            matched_dets.add(det_idx)

        # Create new objects for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                new_obj = TrackedObject(self.next_id, frame, det, confidences[i], cls_ids[i], self.max_batch_size)
                self.objects.append(new_obj)
                self.next_id += 1

        # Finalize expired tracks
        completed_data = []
        remaining_objects = []
        
        for obj in self.objects:
            if not obj.matched_this_cycle:
                obj.missing_frames += 1
                if obj.missing_frames >= self.max_age:
                    obj.has_ended = True
            else:
                obj.missing_frames = 0
            
            obj.matched_this_cycle = False

            if obj.has_ended or obj.needs_flush:
                # Signal data is ready (could be used for database/cloud upload later)
                completed_data.append(obj)
                
                if obj.needs_flush and not obj.has_ended:
                    # Partial flush: Keep track but clear image buffer to save RAM
                    obj.images = []
                    obj.needs_flush = False

            if not obj.has_ended:
                remaining_objects.append(obj)
        
        self.objects = remaining_objects
        return completed_data
