import numpy as np

class TrackedObject:
    def __init__(self, obj_id, bbox, cls_id=None):
        self.id = obj_id
        self.bbox = np.array(bbox)  # [x1, y1, x2, y2]
        self.cls_id = cls_id
        self.velocity = np.zeros(4) # [vx1, vy1, vx2, vy2]
        self.age = 1
        self.missing = 0
        self.saved_count = 0
        self.last_saved_frame = -100
        self.history = [self.bbox.copy()]

    def predict(self):
        """Predict the next bounding box position based on constant velocity."""
        predicted_bbox = self.bbox + self.velocity
        return predicted_bbox

    def update(self, new_bbox, cls_id=None):
        """Update state with a new detection."""
        new_bbox = np.array(new_bbox)
        self.velocity = new_bbox - self.bbox
        self.bbox = new_bbox
        if cls_id is not None:
            self.cls_id = cls_id
        self.missing = 0
        self.age += 1
        self.history.append(self.bbox.copy())
        if len(self.history) > 10:
            self.history.pop(0)

    def mark_missing(self):
        """Handle frames where detection was lost."""
        self.missing += 1
        self.bbox = self.predict() # Move forward by velocity even if lost

class PlateTracker:
    """
    Core Principle: "Predict & Match"
    1. Predict: Calculates where a vehicle should be based on its previous movement (Constant Velocity Model).
    2. Match: Pairs detections with existing tracks using IOU and Dynamic Distance Fallback.
    3. Persistence: Keeps IDs "alive" for max_age frames to handle brief detection losses (ghosting).
    """
    def __init__(self, max_age=30, iou_threshold=0.3, distance_threshold_ratio=0.5):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.distance_threshold_ratio = distance_threshold_ratio

    @staticmethod
    def get_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detections, cls_ids=None):
        """
        detections: list of [x1, y1, x2, y2]
        cls_ids: list of class IDs corresponding to detections
        """
        if cls_ids is None:
            cls_ids = [None] * len(detections)

        if not self.tracks:
            for det, cls_id in zip(detections, cls_ids):
                self.tracks.append(TrackedObject(self.next_id, det, cls_id))
                self.next_id += 1
            return self.tracks

        # 1. Predict future positions
        predictions = [t.predict() for t in self.tracks]
        
        # 2. Calculate Costs (IOU)
        matched_tracks = {} # track_idx -> detection_idx
        used_detections = set()

        # Greedy Match by IOU
        ious = []
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                iou = self.get_iou(pred, det)
                if iou >= self.iou_threshold:
                    ious.append((iou, i, j))
        
        ious.sort(key=lambda x: x[0], reverse=True)
        for iou, t_idx, d_idx in ious:
            if t_idx not in matched_tracks and d_idx not in used_detections:
                matched_tracks[t_idx] = d_idx
                used_detections.add(d_idx)

        # 3. Dynamic Distance Fallback for unmatched
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [j for j in range(len(detections)) if j not in used_detections]

        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            pred_box = predictions[t_idx]
            pred_center = np.array([(pred_box[0] + pred_box[2])/2, 
                                    (pred_box[1] + pred_box[3])/2])
            
            # Dynamic Limit: allowed distance depends on box size (width)
            track_width = track.bbox[2] - track.bbox[0]
            dist_limit = track_width * self.distance_threshold_ratio
            
            best_dist = float('inf')
            best_d_idx = -1
            
            for d_idx in unmatched_detections:
                det = detections[d_idx]
                det_center = np.array([(det[0] + det[2])/2, (det[1] + det[3])/2])
                dist = np.linalg.norm(pred_center - det_center)
                
                if dist < dist_limit and dist < best_dist:
                    best_dist = dist
                    best_d_idx = d_idx
            
            if best_d_idx != -1:
                matched_tracks[t_idx] = best_d_idx
                used_detections.add(best_d_idx)
                # Note: We don't remove from unmatched_detections here because it's handled by used_detections

        # 4. Recover & Move
        for i, track in enumerate(self.tracks):
            if i in matched_tracks:
                d_idx = matched_tracks[i]
                track.update(detections[d_idx], cls_ids[d_idx])
            else:
                track.mark_missing()

        # 5. Spawn new IDs for unused detections
        for d_idx in range(len(detections)):
            if d_idx not in used_detections:
                self.tracks.append(TrackedObject(self.next_id, detections[d_idx], cls_ids[d_idx]))
                self.next_id += 1

        # 6. Cleanup
        self.tracks = [t for t in self.tracks if t.missing <= self.max_age]

        return self.tracks
