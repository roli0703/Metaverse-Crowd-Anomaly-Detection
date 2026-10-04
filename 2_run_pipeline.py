import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

DEPARTURE_RESTRICTED_ZONE = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], np.int32)

def run_detection(video_path, output_path, restricted_zone=None, crowd_radius=140):
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open '{video_path}'. Check file path.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nProcessing '{video_path}' ({total_frames} frames)...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # conf=0.20 enables detection of occluded/partially covered bodies
        # custom_bytetrack.yaml preserves track identity during overlap
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=[0], 
            conf=0.15, 
            iou=0.5,
            verbose=False
        )[0]
        
        if restricted_zone is not None:
            cv2.polylines(frame, [restricted_zone], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, "RESTRICTED ZONE", (110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        person_data = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            
            # Step 1: Detect Individual Actions
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Fall check
                aspect_ratio = w / (h + 1e-6)
                is_fallen = aspect_ratio > 1.18
                
                is_intruder = False
                if restricted_zone is not None:
                    is_intruder = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0
                
                person_data.append({
                    'id': track_id,
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (cx, cy),
                    'is_fallen': is_fallen,
                    'is_intruder': is_intruder
                })

            # Step 2: Accurate Cluster Formation (At least 4 people)
            crowd_cluster_ids = set()
            num_people = len(person_data)
            
            for i in range(num_people):
                cluster_members = [person_data[i]['id']]
                c1 = person_data[i]['center']
                
                for j in range(num_people):
                    if i == j:
                        continue
                    c2 = person_data[j]['center']
                    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    
                    if dist < crowd_radius:
                        cluster_members.append(person_data[j]['id'])
                        
                # Only if at least 4 individuals are in the immediate vicinity
                if len(cluster_members) >= 4:
                    crowd_cluster_ids.update(cluster_members)

            # Step 3: Visual Annotations
            for p in person_data:
                x1, y1, x2, y2 = p['box']
                track_id = p['id']
                cx, cy = p['center']
                
                # Emergency takes priority
                if p['is_fallen']:
                    color = (0, 0, 255)    # Red
                    label = f"EMERGENCY: Fallen ID {track_id}"
                elif p['is_intruder']:
                    color = (0, 69, 255)   # Orange-Red
                    label = f"ALERT: Intruder {track_id}"
                elif track_id in crowd_cluster_ids:
                    color = (0, 255, 255)  # Yellow for valid crowd cluster
                    label = f"CROWD: ID {track_id}"
                else:
                    color = (0, 255, 0)    # Green for normal
                    label = f"ID: {track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

            # Prominent banner only when >= 4 people are clustered
            if len(crowd_cluster_ids) >= 4:
                # Banner background rectangle for high visibility
                cv2.rectangle(frame, (30, 20), (520, 70), (0, 0, 0), -1)
                cv2.putText(frame, f"CROWD ALERT: {len(crowd_cluster_ids)} PEOPLE GATHERED", 
                            (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Done! Video saved to '{output_path}'.")

# Run on Departure
departure_video = Path("Departure") / "07-00-46.mp4"
if departure_video.exists():
    run_detection(
        video_path=departure_video, 
        output_path="output_departure_anomalies.mp4", 
        restricted_zone=DEPARTURE_RESTRICTED_ZONE,
        crowd_radius=150
    )

# Run on Entrance
entrance_files = list(Path("Entrance").glob("*.mp4"))
if entrance_files:
    run_detection(
        video_path=entrance_files[0], 
        output_path="output_entrance_crowd.mp4", 
        restricted_zone=None,
        crowd_radius=150
    )