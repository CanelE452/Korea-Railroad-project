"""gt_manual/ (priority) + gt/ (fallback) 병합 → gt_final/ 생성.

사용법:
    python scripts/data_prep/finalize_gt.py \
        --capture data/pallet/raw_data/capture0403middle
"""
import argparse
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    args = p.parse_args()

    auto_dir = os.path.join(args.capture, "gt")
    manual_dir = os.path.join(args.capture, "gt_manual")
    final_dir = os.path.join(args.capture, "gt_final")

    auto_ov = os.path.join(args.capture, "gt_overlay")
    manual_ov = os.path.join(args.capture, "gt_manual_overlay")
    final_ov = os.path.join(args.capture, "gt_final_overlay")

    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(final_ov, exist_ok=True)

    manual_names = set()
    if os.path.isdir(manual_dir):
        manual_names = {f[:-5] for f in os.listdir(manual_dir) if f.endswith(".json")}

    auto_names = set()
    if os.path.isdir(auto_dir):
        auto_names = {f[:-5] for f in os.listdir(auto_dir) if f.endswith(".json")}

    all_names = manual_names | auto_names
    stats = {"manual": 0, "auto": 0, "none": 0}

    for name in sorted(all_names):
        if name in manual_names:
            # Copy manual
            shutil.copy(os.path.join(manual_dir, f"{name}.json"),
                        os.path.join(final_dir, f"{name}.json"))
            src_ov = os.path.join(manual_ov, f"{name}.jpg")
            if os.path.exists(src_ov):
                shutil.copy(src_ov, os.path.join(final_ov, f"{name}.jpg"))
            stats["manual"] += 1
        elif name in auto_names:
            shutil.copy(os.path.join(auto_dir, f"{name}.json"),
                        os.path.join(final_dir, f"{name}.json"))
            src_ov = os.path.join(auto_ov, f"{name}.jpg")
            if os.path.exists(src_ov):
                shutil.copy(src_ov, os.path.join(final_ov, f"{name}.jpg"))
            stats["auto"] += 1
        else:
            stats["none"] += 1

    print(f"Final GT: {final_dir}")
    print(f"  from manual: {stats['manual']}")
    print(f"  from auto:   {stats['auto']}")
    print(f"  total:       {sum(stats.values())}")


if __name__ == "__main__":
    main()
