"""배치별 생성 데이터를 하나의 train/ 디렉토리로 병합 + 검증.

사용법:
    python scripts/data_prep/validate/merge_and_validate.py
"""

import glob
import json
import os
import shutil
import sys

BASE_DIR = os.path.join("data", "pallet", "training_data")
TRAIN_DIRS = sorted([d for d in os.listdir(os.path.join("data", "pallet", "training_data"))
                     if (d.startswith("train_batch_") or d.startswith("train_v11_batch_") or d.startswith("train_far_batch_"))
                     and os.path.isdir(os.path.join("data", "pallet", "training_data", d))])
VAL_BATCH_DIRS = sorted([d for d in os.listdir(os.path.join("data", "pallet", "training_data"))
                         if (d.startswith("val_batch_") or d.startswith("val_v11_batch_") or d.startswith("val_far_batch_"))
                         and os.path.isdir(os.path.join("data", "pallet", "training_data", d))])
VAL_DIR = "val"
MERGED_TRAIN = "train"

MIN_VISIBILITY = 0.5


def merge_batches():
    """배치 디렉토리들을 train/ 으로 병합 (연속 번호 재부여)."""
    merged_dir = os.path.join(BASE_DIR, MERGED_TRAIN)
    os.makedirs(merged_dir, exist_ok=True)

    global_idx = 0
    for batch_name in TRAIN_DIRS:
        batch_dir = os.path.join(BASE_DIR, batch_name)
        if not os.path.isdir(batch_dir):
            print(f"  [SKIP] {batch_dir} not found")
            continue

        png_files = sorted(glob.glob(os.path.join(batch_dir, "*.png")))
        print(f"  {batch_name}: {len(png_files)} frames")

        for png_path in png_files:
            basename = os.path.splitext(os.path.basename(png_path))[0]
            json_path = os.path.join(batch_dir, basename + ".json")

            if not os.path.exists(json_path):
                continue

            new_name = f"{global_idx:06d}"
            shutil.copy2(png_path, os.path.join(merged_dir, new_name + ".png"))
            shutil.copy2(json_path, os.path.join(merged_dir, new_name + ".json"))
            global_idx += 1

    print(f"\n  Merged: {global_idx} frames -> {merged_dir}")
    return global_idx


def validate_dir(data_dir, label=""):
    """데이터 디렉토리 검증."""
    png_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    png_basenames = {os.path.splitext(os.path.basename(f))[0] for f in png_files}
    json_basenames = {os.path.splitext(os.path.basename(f))[0] for f in json_files}

    paired = png_basenames & json_basenames
    orphan_png = png_basenames - json_basenames
    orphan_json = json_basenames - png_basenames

    errors = 0
    low_vis = 0
    valid = 0

    for name in sorted(paired):
        json_path = os.path.join(data_dir, name + ".json")
        try:
            with open(json_path) as f:
                data = json.load(f)

            for obj in data.get("objects", []):
                if "projected_cuboid" not in obj:
                    errors += 1
                    continue
                cuboid = obj["projected_cuboid"]
                if len(cuboid) != 8:
                    errors += 1
                    continue
                if obj.get("visibility", 0) < MIN_VISIBILITY:
                    low_vis += 1
                else:
                    valid += 1
        except (json.JSONDecodeError, KeyError) as e:
            errors += 1

    print(f"\n  [{label}] {data_dir}")
    print(f"    PNG: {len(png_files)}, JSON: {len(json_files)}")
    print(f"    Paired: {len(paired)}")
    print(f"    Valid (vis>={MIN_VISIBILITY}): {valid}")
    print(f"    Low visibility: {low_vis}")
    print(f"    Errors: {errors}")
    print(f"    Orphan PNG: {len(orphan_png)}, Orphan JSON: {len(orphan_json)}")

    return valid, errors


def main():
    print("=" * 50)
    print(" Merge & Validate Synthetic Data")
    print("=" * 50)

    # 1. 병합
    print("\n[Step 1] Merging train batches...")
    total = merge_batches()

    # 1b. Val 배치 병합
    print("\n[Step 1b] Merging val batches...")
    val_dir = os.path.join(BASE_DIR, VAL_DIR)
    os.makedirs(val_dir, exist_ok=True)
    val_idx = 0
    for batch_name in VAL_BATCH_DIRS:
        batch_dir = os.path.join(BASE_DIR, batch_name)
        if not os.path.isdir(batch_dir):
            continue
        png_files = sorted(glob.glob(os.path.join(batch_dir, "*.png")))
        print(f"  {batch_name}: {len(png_files)} frames")
        for png_path in png_files:
            basename = os.path.splitext(os.path.basename(png_path))[0]
            json_path = os.path.join(batch_dir, basename + ".json")
            if not os.path.exists(json_path):
                continue
            new_name = f"{val_idx:06d}"
            shutil.copy2(png_path, os.path.join(val_dir, new_name + ".png"))
            shutil.copy2(json_path, os.path.join(val_dir, new_name + ".json"))
            val_idx += 1
    print(f"  Merged: {val_idx} val frames -> {val_dir}")

    # 2. 검증
    print("\n[Step 2] Validating...")
    train_dir = os.path.join(BASE_DIR, MERGED_TRAIN)

    train_valid, train_err = validate_dir(train_dir, "TRAIN")
    val_valid, val_err = 0, 0
    if os.path.isdir(val_dir):
        val_valid, val_err = validate_dir(val_dir, "VAL")

    # 3. 요약
    print("\n" + "=" * 50)
    print(f" Train: {train_valid} valid frames")
    print(f" Val:   {val_valid} valid frames")
    print(f" Total errors: {train_err + val_err}")
    print("=" * 50)

    if train_err + val_err > 0:
        print("\n[WARN] Errors found. Check data before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
