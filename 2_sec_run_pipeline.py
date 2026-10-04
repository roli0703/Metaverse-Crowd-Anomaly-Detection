import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 1. Initialize detector model
detector = YOLO('yolov8n.pt')

# Restricted Zone strictly for Departure Terminal
DEPARTURE_RESTRICTED_ZONE = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], np.int32)

def run_deepsort_stream(video_path, output_path, restricted_zone=None, crowd_radius=150):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open '{video_path}'. Check file path.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize DeepSORT Tracker
    tracker = DeepSort(
        max_age=30,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3
    )

    print(f"\nProcessing '{video_path}' -> Saving to '{output_path}' ({total_frames} frames)...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step A: YOLOv8 person detection
        results = detector(frame, classes=[0], conf=0.20, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                detections.append(([int(x1), int(y1), int(w), int(h)], float(conf), 'person'))

        # Step B: DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw Restricted Zone only if specified
        if restricted_zone is not None:
            cv2.polylines(frame, [restricted_zone], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, "RESTRICTED ZONE", (110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        person_data = []

        # Step C: Track verification and anomaly checks
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            w = x2 - x1
            h = y2 - y1
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Fall detection check
            aspect_ratio = w / (h + 1e-6)
            is_fallen = aspect_ratio > 1.18

            # Geofence intrusion check
            is_intruder = False
            if restricted_zone is not None:
                is_intruder = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0

            person_data.append({
                'id': track_id,
                'box': (x1, y1, x2, y2),
                'center': (cx, cy),
                'is_fallen': is_fallen,
                'is_intruder': is_intruder
            })

        # Step D: Spatial clustering for crowd gathering (>= 4 individuals)
        crowd_cluster_ids = set()
        num_people = len(person_data)
        for i in range(num_people):
            cluster_members = [person_data[i]['id']]
            c1 = person_data[i]['center']
            for j in range(num_people):
                if i == j:
                    continue
                c2 = person_data[j]['center']
                if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < crowd_radius:
                    cluster_members.append(person_data[j]['id'])

            if len(cluster_members) >= 4:
                crowd_cluster_ids.update(cluster_members)

        # Step E: Visual rendering
        for p in person_data:
            x1, y1, x2, y2 = p['box']
            track_id = p['id']
            cx, cy = p['center']

            if p['is_fallen']:
                color = (0, 0, 255)
                label = f"EMERGENCY: Fallen ID {track_id}"
            elif p['is_intruder']:
                color = (0, 69, 255)
                label = f"ALERT: Intruder {track_id}"
            elif track_id in crowd_cluster_ids:
                color = (0, 255, 255)
                label = f"CROWD: ID {track_id}"
            else:
                color = (0, 255, 0)
                label = f"ID: {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # Top banner for high-density gathering
        if len(crowd_cluster_ids) >= 4:
            cv2.rectangle(frame, (30, 20), (530, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"CROWD ALERT: {len(crowd_cluster_ids)} PEOPLE GATHERED", 
                        (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Done! DeepSORT video written to: {output_path}")

# --- Pipeline Execution ---
if __name__ == "__main__":
    # 1. Process Departure
    departure_video = Path("Departure") / "07-00-46.mp4"
    if departure_video.exists():
        run_deepsort_stream(
            video_path=departure_video,
            output_path="output_departure_deepsort.mp4",
            restricted_zone=DEPARTURE_RESTRICTED_ZONE,
            crowd_radius=150
        )

    # 2. Process Entrance
    entrance_files = list(Path("Entrance").glob("*.mp4"))
    if entrance_files:
        run_deepsort_stream(
            video_path=entrance_files[0],
            output_path="output_entrance_deepsort.mp4",
            restricted_zone=None,
            crowd_radius=150
        )