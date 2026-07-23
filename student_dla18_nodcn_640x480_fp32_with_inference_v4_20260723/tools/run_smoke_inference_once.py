#!/usr/bin/env python3
"""Run one SMOKE checkpoint on a KITTI-layout split and zip predictions."""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, default=Path("SMOKE-master"))
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-subdir", default="testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-kitti-eval", action="store_true")
    parser.add_argument("--zip-name", default=None)
    return parser.parse_args()


def ensure_symlink(link_path, target_path):
    link_path = Path(link_path)
    target_path = Path(target_path).resolve()
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f"Refusing to replace non-symlink: {link_path}")
    link_path.symlink_to(target_path)


def prediction_counts(output_dir):
    data_dir = Path(output_dir) / "inference" / "kitti_test" / "data"
    files = sorted(data_dir.glob("*.txt"))
    nonempty = sum(1 for path in files if path.stat().st_size > 0)
    return len(files), nonempty, data_dir


def zip_predictions(data_dir, zip_path):
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(data_dir).glob("*.txt")):
            archive.write(path, Path("data") / path.name)
    return zip_path


def main():
    args = parse_args()
    smoke_dir = args.smoke_dir.resolve()
    source_split_dir = (args.dataset_root / args.dataset_subdir).resolve()
    split_file = source_split_dir / "ImageSets" / f"{args.split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    output_dir = args.output_dir.resolve()
    runtime_kitti = output_dir / "_runtime_kitti"
    runtime_kitti.mkdir(parents=True, exist_ok=True)
    ensure_symlink(runtime_kitti / "testing", source_split_dir)

    smoke_datasets = smoke_dir / "datasets"
    smoke_datasets.mkdir(exist_ok=True)
    ensure_symlink(smoke_datasets / "kitti", runtime_kitti)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{smoke_dir}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(smoke_dir)
    env["SMOKE_SKIP_KITTI_EVAL"] = "1" if args.skip_kitti_eval else "0"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_inference_stdout.log"
    cmd = [
        args.python_bin,
        "tools/plain_train_net.py",
        "--eval-only",
        "--config-file",
        args.config_file,
        "--ckpt",
        str(Path(args.ckpt).resolve()),
        "DATASETS.TEST_SPLIT",
        args.split,
        "DATASETS.DETECT_CLASSES",
        "('Car',)",
        "MODEL.DEVICE",
        args.device,
        "TEST.FP16",
        "True" if args.fp16 else "False",
        "OUTPUT_DIR",
        str(output_dir),
    ]

    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=smoke_dir, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(f"Inference failed; see {log_path}")

    files, nonempty, data_dir = prediction_counts(output_dir)
    zip_name = args.zip_name or f"{args.split}_predictions_{Path(args.ckpt).stem}_{'fp16' if args.fp16 else 'fp32'}.zip"
    zip_path = zip_predictions(data_dir, output_dir / "inference" / "kitti_test" / zip_name)
    meta = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "config_file": args.config_file,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_subdir": args.dataset_subdir,
        "split": args.split,
        "fp16": args.fp16,
        "skip_kitti_eval": args.skip_kitti_eval,
        "output_dir": str(output_dir),
        "prediction_dir": str(data_dir),
        "prediction_zip": str(zip_path),
        "prediction_files": files,
        "nonempty_prediction_files": nonempty,
        "elapsed_seconds": elapsed,
        "stdout_log": str(log_path),
    }
    (output_dir / "inference_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"INFERENCE_DONE files={files} nonempty={nonempty} elapsed={elapsed:.1f}s "
        f"fp16={args.fp16} zip={zip_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
