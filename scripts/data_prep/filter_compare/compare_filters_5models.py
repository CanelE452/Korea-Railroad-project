"""v8 vs ablation A/B/C/D - 5 model filter comparison.

Each filter (A, B, C) -> union of all passed images -> 5-column side-by-side.
"""
import csv
import os
import cv2
import numpy as np


MODELS = [
    ("v8", "data/pallet/eval_results/v8_noapril"),
    ("ablA_coord", "data/pallet/eval_results/v8ablationa_noapril"),
    ("ablB_edge", "data/pallet/eval_results/v8ablationb_noapril"),
    ("ablC_co+ed", "data/pallet/eval_results/v8ablationc_noapril"),
    ("ablD_flip", "data/pallet/eval_results/v8ablationd_noapril"),
]

FILTERS = ["A", "B", "C"]
OUT_BASE = "data/pallet/eval_results/ablation_5model_compare"


def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["filename"]] = row
    return rows


def get_score_label(filter_name, row):
    if not row:
        return ""
    if filter_name == "A":
        return f"sA={row.get('sA_score', '?')}"
    elif filter_name == "B":
        return f"sp={row.get('span', '?')}"
    elif filter_name == "C":
        return f"sC={row.get('sC', '?')}"
    return ""


def make_row_image(images, labels, target_h=300):
    """N images side by side with labels on top."""
    n = len(images)
    # Determine uniform size
    valid_imgs = [img for img in images if img is not None]
    if not valid_imgs:
        return None

    ref = valid_imgs[0]
    aspect = ref.shape[1] / ref.shape[0]
    cell_w = int(target_h * aspect)
    cell_h = target_h

    bar_h = 36
    sep_w = 4
    total_w = n * cell_w + (n - 1) * sep_w
    total_h = bar_h + cell_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for i in range(n):
        x_off = i * (cell_w + sep_w)

        # Label bar
        cv2.putText(canvas, labels[i], (x_off + 4, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Image
        if images[i] is not None:
            resized = cv2.resize(images[i], (cell_w, cell_h))
            canvas[bar_h:bar_h + cell_h, x_off:x_off + cell_w] = resized
        else:
            # Dark placeholder
            cv2.putText(canvas, "no overlay", (x_off + cell_w // 4, bar_h + cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)

    return canvas


def main():
    # Load all CSVs
    csvs = {}
    for name, dirpath in MODELS:
        csvs[name] = load_csv(os.path.join(dirpath, "filter_details.csv"))

    for filt in FILTERS:
        out_dir = os.path.join(OUT_BASE, filt)
        os.makedirs(out_dir, exist_ok=True)

        # Collect union of all filenames that passed this filter in any model
        all_fnames = set()
        for name, dirpath in MODELS:
            filt_dir = os.path.join(dirpath, filt)
            if os.path.isdir(filt_dir):
                for f in os.listdir(filt_dir):
                    if f.endswith(".jpg"):
                        all_fnames.add(f.replace("_overlay.jpg", ""))

        all_fnames = sorted(all_fnames)
        if not all_fnames:
            print(f"  {filt}: no images from any model")
            continue

        count = 0
        for fname in all_fnames:
            images = []
            labels = []
            for mname, dirpath in MODELS:
                # Try filter-specific overlay first, then 'all'
                filt_path = os.path.join(dirpath, filt, f"{fname}_overlay.jpg")
                all_path = os.path.join(dirpath, "all", f"{fname}_overlay.jpg")

                passed = os.path.exists(filt_path)
                img = cv2.imread(filt_path) if passed else cv2.imread(all_path)
                images.append(img)

                tag = "PASS" if passed else "FAIL"
                score = get_score_label(filt, csvs[mname].get(fname, {}))
                labels.append(f"{mname} [{tag}] {score}")

            result = make_row_image(images, labels)
            if result is not None:
                cv2.imwrite(os.path.join(out_dir, f"{fname}.jpg"), result)
                count += 1

        # Summary per filter
        per_model = []
        for mname, dirpath in MODELS:
            filt_dir = os.path.join(dirpath, filt)
            n = len([f for f in os.listdir(filt_dir) if f.endswith(".jpg")]) if os.path.isdir(filt_dir) else 0
            per_model.append(f"{mname}={n}")
        print(f"  Filter {filt}: {count} comparisons ({', '.join(per_model)})")

    print(f"\nOutput: {OUT_BASE}/")


if __name__ == "__main__":
    main()
