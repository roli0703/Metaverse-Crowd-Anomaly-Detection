import cv2
import os
from pathlib import Path

# Subfolders ke paths define karein
scenes = ["Departure", "Entrance"]
output_dir = Path("dataset_frames")
output_dir.mkdir(exist_ok=True)

TARGET_FPS = 10  # Section 3.6 के अनुसार 10 FPS

for scene in scenes:
    scene_path = Path(scene)
    if not scene_path.exists():
        print(f"Folder nahi mila: {scene}")
        continue

    for video_file in scene_path.glob("*.mp4"):
        cap = cv2.VideoCapture(str(video_file))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(max(1, round(native_fps / TARGET_FPS))) if native_fps > 0 else 1
        
        save_folder = output_dir / scene / video_file.stem
        save_folder.mkdir(parents=True, exist_ok=True)
        
        frame_count, saved_count = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                cv2.imwrite(str(save_folder / f"frame_{saved_count:05d}.jpg"), frame)
                saved_count += 1
            frame_count += 1
            
        cap.release()
        print(f"Done: {scene}/{video_file.name} se {saved_count} frames extract ho gaye.")