#!/usr/bin/env python3
"""Benchmark SMOKE model forward speed on a fixed-size synthetic input."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, default=Path("SMOKE-master"))
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    smoke_dir = args.smoke_dir.resolve()
    sys.path.insert(0, str(smoke_dir))

    from smoke.config import cfg as base_cfg
    from smoke.modeling.detector import build_detection_model
    from smoke.utils.check_point import DetectronCheckpointer

    cfg = base_cfg.clone()
    cfg.merge_from_file(str(smoke_dir / args.config_file))
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    device = torch.device(args.device)
    model = build_detection_model(cfg)
    model.to(device)
    DetectronCheckpointer(cfg, model, save_dir="").load(str(Path(args.ckpt).resolve()), use_latest=False)
    model.eval()

    x = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.amp.autocast(device_type=device.type, enabled=args.fp16 and device.type == "cuda"):
                model.backbone(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            with torch.amp.autocast(device_type=device.type, enabled=args.fp16 and device.type == "cuda"):
                features = model.backbone(x)
                model.heads.predictor(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    result = {
        "config_file": args.config_file,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "device": args.device,
        "input_size": [args.width, args.height],
        "warmup": args.warmup,
        "iters": args.iters,
        "fp16": args.fp16,
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / args.iters,
        "fps": args.iters / elapsed,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
