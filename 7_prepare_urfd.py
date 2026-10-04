import os
import shutil
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent
UR_DATA_DIR = BASE_DIR / "UR_data"
OUTPUT_DIR = BASE_DIR / "data" / "processed_fall_dataset"

# Target directories
TRAIN_FALL = OUTPUT_DIR / "train" / "fall"
TRAIN_NORMAL = OUTPUT_DIR / "train" / "normal"
VAL_FALL = OUTPUT_DIR / "val" / "fall"
VAL_NORMAL = OUTPUT_DIR / "val" / "normal"

for p in [TRAIN_FALL, TRAIN_NORMAL, VAL_FALL, VAL_NORMAL]:
    p.mkdir(parents=True, exist_ok=True)

# Validation sequences
VAL_FALL_SEQS = ["fall-25", "fall-26", "fall-27", "fall-28", "fall-29", "fall-30"]
VAL_ADL_SEQS = ["adl-35", "adl-36", "adl-37", "adl-38", "adl-39", "adl-40"]

print(f"Scanning files inside: {UR_DATA_DIR.resolve()}")

fall_count = 0
normal_count = 0

# UR_data ke andar jitne bhi files hain sabko recursively scan karein
for file_path in UR_DATA_DIR.rglob("*"):
    if file_path.is_file() and file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        full_path_str = str(file_path).lower()

        # Check agar path mein 'fall' hai
        if "fall" in full_path_str:
            is_val = any(seq in full_path_str for seq in VAL_FALL_SEQS)
            dest = VAL_FALL if is_val else TRAIN_FALL
            dest_name = f"{file_path.parent.name}_{file_path.name}"
            shutil.copy2(file_path, dest / dest_name)
            fall_count += 1

        # Check agar path mein 'adl' hai (Normal)
        elif "adl" in full_path_str:
            is_val = any(seq in full_path_str for seq in VAL_ADL_SEQS)
            dest = VAL_NORMAL if is_val else TRAIN_NORMAL
            dest_name = f"{file_path.parent.name}_{file_path.name}"
            shutil.copy2(file_path, dest / dest_name)
            normal_count += 1

print(f"\nDone! Copied:")
print(f"  - Fall images: {fall_count}")
print(f"  - Normal (ADL) images: {normal_count}")
print(f"Dataset ready at: {OUTPUT_DIR}")