"""v8 vs v8ablationa 필터 비교 이미지 생성.

각 필터(A, B, C)별로 두 모델의 overlay를 나란히 비교.
"""
import csv
import os
import cv2
import numpy as np


def load_csv(path):
    """CSV를 dict(filename -> row)로 로드."""
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["filename"]] = row
    return rows


def make_side_by_side(img_left, img_right, label_left, label_right):
    """두 이미지를 나란히 배치 + 라벨."""
    if img_left is None and img_right is None:
        return None

    h = max(
        img_left.shape[0] if img_left is not None else 0,
        img_right.shape[0] if img_right is not None else 0,
    )
    w = max(
        img_left.shape[1] if img_left is not None else 0,
        img_right.shape[1] if img_right is not None else 0,
    )

    if img_left is None:
        img_left = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img_left, "N/A", (w // 3, h // 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (80, 80, 80), 3)
    if img_right is None:
        img_right = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img_right, "N/A", (w // 3, h // 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (80, 80, 80), 3)

    # Resize to same height
    img_left = cv2.resize(img_left, (w, h))
    img_right = cv2.resize(img_right, (w, h))

    # Header bar
    bar_h = 40
    bar = np.zeros((bar_h, w * 2 + 10, 3), dtype=np.uint8)
    cv2.putText(bar, label_left, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
    cv2.putText(bar, label_right, (w + 20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 200), 2)

    # Separator
    sep = np.ones((h, 10, 3), dtype=np.uint8) * 40
    combined = np.hstack([img_left, sep, img_right])
    result = np.vstack([bar, combined])
    return result


def compare_filter(filter_name, dir_v8, dir_abla, csv_v8, csv_abla, out_dir):
    """한 필터에 대해 두 모델 비교 이미지 생성."""
    os.makedirs(out_dir, exist_ok=True)

    # Collect all filenames from both
    v8_files = set()
    abla_files = set()

    v8_filter_dir = os.path.join(dir_v8, filter_name)
    abla_filter_dir = os.path.join(dir_abla, filter_name)

    if os.path.isdir(v8_filter_dir):
        for f in os.listdir(v8_filter_dir):
            if f.endswith(".jpg"):
                v8_files.add(f.replace("_overlay.jpg", ""))
    if os.path.isdir(abla_filter_dir):
        for f in os.listdir(abla_filter_dir):
            if f.endswith(".jpg"):
                abla_files.add(f.replace("_overlay.jpg", ""))

    all_files = sorted(v8_files | abla_files)

    if not all_files:
        print(f"  {filter_name}: no images")
        return

    # Generate comparison for union of both
    count = 0
    for fname in all_files:
        v8_path = os.path.join(v8_filter_dir, f"{fname}_overlay.jpg")
        abla_path = os.path.join(abla_filter_dir, f"{fname}_overlay.jpg")

        img_v8 = cv2.imread(v8_path) if os.path.exists(v8_path) else None
        img_abla = cv2.imread(abla_path) if os.path.exists(abla_path) else None

        in_v8 = "PASS" if fname in v8_files else "FAIL"
        in_abla = "PASS" if fname in abla_files else "FAIL"

        # Get scores from CSV
        v8_row = csv_v8.get(fname, {})
        abla_row = csv_abla.get(fname, {})

        label_v8 = f"v8 [{in_v8}]"
        label_abla = f"v8ablationA [{in_abla}]"

        if filter_name == "A":
            s_v8 = v8_row.get("sA_score", "?")
            s_abla = abla_row.get("sA_score", "?")
            label_v8 += f" sA={s_v8}"
            label_abla += f" sA={s_abla}"
        elif filter_name == "B":
            s_v8 = v8_row.get("span", "?")
            s_abla = abla_row.get("span", "?")
            label_v8 += f" span={s_v8}"
            label_abla += f" span={s_abla}"
        elif filter_name == "C":
            s_v8 = v8_row.get("sC", "?")
            s_abla = abla_row.get("sC", "?")
            label_v8 += f" sC={s_v8}"
            label_abla += f" sC={s_abla}"

        # Use 'all' overlay if filter-specific one is missing
        if img_v8 is None:
            fallback = os.path.join(dir_v8, "all", f"{fname}_overlay.jpg")
            if os.path.exists(fallback):
                img_v8 = cv2.imread(fallback)
        if img_abla is None:
            fallback = os.path.join(dir_abla, "all", f"{fname}_overlay.jpg")
            if os.path.exists(fallback):
                img_abla = cv2.imread(fallback)

        result = make_side_by_side(img_v8, img_abla, label_v8, label_abla)
        if result is not None:
            cv2.imwrite(os.path.join(out_dir, f"{fname}_compare.jpg"), result)
            count += 1

    print(f"  {filter_name}: {count} comparisons (v8={len(v8_files)}, ablA={len(abla_files)})")


def main():
    base = "data/pallet/eval_results"
    dir_v8 = os.path.join(base, "v8_noapril")
    dir_abla = os.path.join(base, "v8ablationa_noapril")
    out_base = os.path.join(base, "v8_vs_v8ablationa_compare")

    csv_v8 = load_csv(os.path.join(dir_v8, "filter_details.csv"))
    csv_abla = load_csv(os.path.join(dir_abla, "filter_details.csv"))

    print(f"v8:        PnP={sum(1 for r in csv_v8.values() if r['pnp_ok']=='True')}")
    print(f"v8ablA:    PnP={sum(1 for r in csv_abla.values() if r['pnp_ok']=='True')}")
    print()

    for filt in ["A", "B", "C"]:
        compare_filter(filt, dir_v8, dir_abla, csv_v8, csv_abla,
                        os.path.join(out_base, filt))

    print(f"\nOutput: {out_base}/")


if __name__ == "__main__":
    main()
