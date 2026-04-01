import numpy as np
from utils.tracker import PlateTracker

def test_tracker():
    tracker = PlateTracker(max_age=5, iou_threshold=0.3, distance_threshold_ratio=0.5)
    
    # Simulate a vehicle moving horizontally
    # Frame 1: Detection at [10, 10, 50, 50], Class 2 (car)
    print("--- Frame 1 ---")
    detections = [[10, 10, 50, 50]]
    cls_ids = [2]
    tracks = tracker.update(detections, cls_ids)
    for t in tracks:
        print(f"Track ID: {t.id}, BBox: {t.bbox}, Cls: {t.cls_id}, Saved: {t.saved_count}")
        t.saved_count += 1 # Simulate saving
    
    # Frame 2: Detection at [15, 10, 55, 50]
    print("\n--- Frame 2 ---")
    detections = [[15, 10, 55, 50]]
    cls_ids = [2]
    tracks = tracker.update(detections, cls_ids)
    for t in tracks:
        print(f"Track ID: {t.id}, BBox: {t.bbox}, Saved: {t.saved_count}")
        t.saved_count += 1 # Simulate saving

    # Frame 3: Detection at [20, 10, 60, 50]
    print("\n--- Frame 3 ---")
    detections = [[20, 10, 60, 50]]
    cls_ids = [2]
    tracks = tracker.update(detections, cls_ids)
    for t in tracks:
        print(f"Track ID: {t.id}, BBox: {t.bbox}, Saved: {t.saved_count}")
        t.saved_count += 1 # Simulate saving

    assert tracks[0].saved_count == 3
    print("\nTracker Sample Counting Verification Successful!")

if __name__ == "__main__":
    test_tracker()
