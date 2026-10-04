import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "vit_fall_detection_model.pth"
YOLO_WEIGHTS = BASE_DIR / "yolov8n.pt"

FRAMES_DIR = BASE_DIR / "dataset_frames" / "Departure" / "07-00-46"
OUTPUT_DIR = BASE_DIR / "metaverse_fall_predictions" / "vit_07-00-46_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. ViT Model Architecture
def load_vit_model():
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

vit_model = load_vit_model()
yolo = YOLO(str(YOLO_WEIGHTS))

# ViT Transformation (224x224)
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['FALL', 'NORMAL']

def run_vit_inference():
    image_paths = sorted([
        p for p in FRAMES_DIR.glob("*.*") 
        if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ])
    
    total_frames = len(image_paths)
    print(f"Total frames found: {total_frames}")

    fall_detected_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w, _ = frame.shape
        has_fall = False

        # Detect person using YOLO
        results = yolo(frame, classes=[0], verbose=False)

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                crop = frame[y1:y2, x1:x2]
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(rgb_crop)
                input_tensor = vit_transform(pil_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = vit_model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)

                label = classes[pred_idx.item()]
                conf = confidence.item() * 100
                aspect_ratio = (x2 - x1) / float(y2 - y1)

                # Fall Decision
                if label == 'FALL' and (aspect_ratio > 0.85 or conf > 90.0):
                    has_fall = True
                    color = (0, 0, 255)
                    tag = f"ViT FALL {conf:.1f}%"
                else:
                    color = (0, 255, 0)
                    tag = f"Normal {conf:.1f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if has_fall:
            fall_detected_count += 1
            cv2.rectangle(frame, (20, 20), (460, 75), (0, 0, 200), -1)
            cv2.putText(frame, "ALERT: PERSON FALL DETECTED", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imwrite(str(OUTPUT_DIR / img_path.name), frame)

        if idx % 50 == 0 or idx == total_frames:
            status = "FALL DETECTED" if has_fall else "NORMAL"
            print(f"[{idx}/{total_frames}] {img_path.name} --> {status}")

    print("\n" + "="*50)
    print("ViT Inference Complete!")
    print(f"Total Frames Evaluated : {total_frames}")
    print(f"Frames with Fall Alert : {fall_detected_count}")
    print(f"Saved Results to       : {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    run_vit_inference()