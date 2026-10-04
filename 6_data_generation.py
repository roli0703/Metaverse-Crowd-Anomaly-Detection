import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> Using Device: {device}")

# 2. YOLOv8 Person Detector
detector = YOLO('yolov8n.pt')
detector.to(device)

# 3. Vision Transformer Setup (Official Torchvision ViT - No HF/CAS errors)
print("--> Loading Vision Transformer (ViT-B/16)...")
vit_weights = models.ViT_B_16_Weights.DEFAULT
vit_model = models.vit_b_16(weights=vit_weights)
vit_model.heads = torch.nn.Identity()  # 768-dim raw feature embeddings
vit_model = vit_model.to(device)
vit_model.eval()

# 4. ViT Transform Pipeline
vit_transform = vit_weights.transforms()

# 5. Non-Destructive Dynamic Folder Setup (Purana dataset delete nahi hoga)
def get_unique_run_dir(base_name="dataset_vit_run"):
    counter = 1
    while True:
        run_path = Path(f"{base_name}_{counter}")
        if not run_path.exists():
            return run_path
        counter += 1

RUN_DIR = get_unique_run_dir()
NORMAL_DIR = RUN_DIR / "normal"
ABNORMAL_DIR = RUN_DIR / "abnormal"

NORMAL_DIR.mkdir(parents=True, exist_ok=True)
ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"--> Target Dataset Directory: {RUN_DIR}")

# Ground truth verified abnormal track ID
VERIFIED_ABNORMAL_IDS = {67}

def process_video_pipeline(video_path, frame_stride=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Video open nahi ho saki at {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"--> Processing Video: {video_path} ({total_frames} total frames)")

    frame_idx = 0
    records = []

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                # YOLOv8 Built-in Tracking
                results = detector.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=0.3,
                    verbose=False,
                    device=device
                )[0]

                if results.boxes is not None and results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    track_ids = results.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        w, h = x2 - x1, y2 - y1
                        if w < 20 or h < 20:
                            continue

                        # Person crop nikalna
                        crop = frame[y1:y2, x1:x2]
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(rgb_crop)

                        # ViT Feature Extraction
                        input_tensor = vit_transform(pil_crop).unsqueeze(0).to(device)
                        features = vit_model(input_tensor)
                        feature_norm = torch.norm(features, p=2).item()

                        # Ground truth rules (Sirf ID 67 abnormal hai)
                        is_abnormal = (track_id in VERIFIED_ABNORMAL_IDS)
                        status = "abnormal" if is_abnormal else "normal"
                        label = 1 if is_abnormal else 0

                        # Dedicated folder save
                        target_dir = (ABNORMAL_DIR if is_abnormal else NORMAL_DIR) / f"id_{track_id}"
                        target_dir.mkdir(parents=True, exist_ok=True)

                        img_name = f"frame_{frame_idx:05d}.jpg"
                        img_path = target_dir / img_name
                        cv2.imwrite(str(img_path), crop)

                        records.append({
                            'frame_idx': frame_idx,
                            'track_id': track_id,
                            'status': status,
                            'label': label,
                            'image_path': str(img_path),
                            'vit_feature_norm': round(feature_norm, 4)
                        })

                        # Bounding Box Draw
                        color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                        label_str = f"ID:{track_id} ({status})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label_str, (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow("Surveillance Pipeline (Press 'q' to exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    output_csv = RUN_DIR / "dataset_manifest.csv"
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nProcessing Complete!")
    print(f"--> Saved frames & manifest to: {RUN_DIR}")
    print(f"--> Total entries: {len(df)}")

if __name__ == "__main__":
    video_path = Path("Departure") / "07-00-46.mp4"
    if not video_path.exists():
        video_path = Path("07-00-46.mp4")

    process_video_pipeline(video_path, frame_stride=2)