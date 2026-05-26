"""restore_orig.py — .orig 백업으로 JSON 복원."""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "data/pallet/training_data/mixed_v8_train",
        "challenge/data/training/v1",
        "challenge/data/training/v2",
    ])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    n_restored = 0
    n_no_orig = 0
    for r in args.roots:
        root = r if os.path.isabs(r) else os.path.join(repo_root, r)
        for orig in glob.glob(os.path.join(root, "**", "*.json.orig"), recursive=True):
            target = orig[:-5]   # strip .orig
            if args.dry_run:
                print(f"  [DRY] would restore: {orig} → {target}")
            else:
                shutil.copy2(orig, target)
                os.remove(orig)
            n_restored += 1
    print(f"\n[Done] restored {n_restored} files {'(DRY)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
