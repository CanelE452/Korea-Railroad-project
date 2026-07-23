#!/usr/bin/env python3
"""Live camera + TFmini-S height input for the bundled geometry_v2 model."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import serial
from serial.tools import list_ports
import torch
from PIL import Image


BUNDLE_DIR = Path(__file__).resolve().parent
SMOKE_DIR = BUNDLE_DIR / "SMOKE-master"
DEFAULT_CONFIG = (
    SMOKE_DIR
    / "configs"
    / "smoke_geometry_v2_real_v4_distill_dla18_nodcn_640x480.yaml"
)
DEFAULT_WEIGHTS = BUNDLE_DIR / "weights" / "model_final.pth"
DEFAULT_K = np.array(
    [
        [605.906494141, 0.0, 317.596191406],
        [0.0, 605.969787598, 256.292297363],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DEFAULT_DIMS_LHW = np.array([5.15, 2.0, 1.75], dtype=np.float32)
MODEL_WIDTH = 640
MODEL_HEIGHT = 480

sys.path.insert(0, str(SMOKE_DIR))

from smoke.config import cfg as smoke_cfg  # noqa: E402
from smoke.data.transforms import build_transforms  # noqa: E402
from smoke.modeling.detector import build_detection_model  # noqa: E402
from smoke.modeling.heatmap_coder import get_transfrom_matrix  # noqa: E402
from smoke.structures.params_3d import ParamsList  # noqa: E402


@dataclass(frozen=True)
class LaserSample:
    distance_m: float
    strength: Optional[int]
    timestamp: float
    raw: str


@dataclass
class Detection:
    score: float
    alpha: float
    bbox: np.ndarray
    dimensions_hwl: np.ndarray
    location_xyz: np.ndarray
    rotation_y: float


class LaserReader:
    _TFMINI_PATTERN = re.compile(
        r"\bL(?P<channel>[12])\s+"
        r"(?P<distance>-?\d+(?:\.\d+)?)\s*cm\b"
        r"(?:\s+strength=(?P<strength>\d+))?",
        re.IGNORECASE,
    )
    _NAMED_PATTERN = re.compile(
        r"(?:distance|dist|range)\s*[:=]\s*"
        r"(?P<distance>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)\b",
        re.IGNORECASE,
    )
    _BARE_PATTERN = re.compile(r"^\s*(?P<distance>\d+(?:\.\d+)?)\s*$")

    def __init__(
        self,
        port: str,
        baud: int,
        channel: int,
        min_distance_m: float,
        max_distance_m: float,
        min_strength: int,
        median_window: int,
    ) -> None:
        self.requested_port = port
        self.baud = baud
        self.channel = channel
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.min_strength = min_strength
        self._values: deque[float] = deque(maxlen=max(1, median_window))
        self._latest: Optional[LaserSample] = None
        self._last_line = ""
        self._error = ""
        self._port_name = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def discover_port(requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        ports = list(list_ports.comports())
        if not ports:
            raise RuntimeError("No serial ports found")
        preferred = [
            p
            for p in ports
            if any(
                token in f"{p.description} {p.manufacturer} {p.hwid}".lower()
                for token in ("ch340", "1a86:7523", "usb-serial", "cp210")
            )
        ]
        return (preferred or ports)[0].device

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="laser-reader",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self, stale_after_s: float) -> tuple[Optional[LaserSample], str, str]:
        with self._lock:
            sample = self._latest
            last_line = self._last_line
            error = self._error
            values = tuple(self._values)
        if sample is None or time.monotonic() - sample.timestamp > stale_after_s:
            return None, last_line, error
        filtered = float(np.median(values)) if values else sample.distance_m
        return (
            LaserSample(
                distance_m=filtered,
                strength=sample.strength,
                timestamp=sample.timestamp,
                raw=sample.raw,
            ),
            last_line,
            error,
        )

    def reset_filter(self) -> None:
        with self._lock:
            self._values.clear()

    def _parse_line(self, line: str) -> Optional[LaserSample]:
        for match in self._TFMINI_PATTERN.finditer(line):
            if int(match.group("channel")) != self.channel:
                continue
            distance_m = float(match.group("distance")) / 100.0
            strength_text = match.group("strength")
            strength = int(strength_text) if strength_text is not None else None
            return self._validated_sample(distance_m, strength, line)

        match = self._NAMED_PATTERN.search(line)
        if match:
            distance = float(match.group("distance"))
            unit = match.group("unit").lower()
            distance_m = distance / 1000.0 if unit == "mm" else distance
            if unit == "cm":
                distance_m /= 100.0
            return self._validated_sample(distance_m, None, line)

        match = self._BARE_PATTERN.match(line)
        if match:
            # Bare TFmini/Arduino values are conventionally centimeters.
            return self._validated_sample(
                float(match.group("distance")) / 100.0,
                None,
                line,
            )
        return None

    def _validated_sample(
        self,
        distance_m: float,
        strength: Optional[int],
        raw: str,
    ) -> Optional[LaserSample]:
        if not self.min_distance_m <= distance_m <= self.max_distance_m:
            return None
        if strength is not None and strength < self.min_strength:
            return None
        return LaserSample(distance_m, strength, time.monotonic(), raw)

    def _run(self) -> None:
        try:
            port_name = self.discover_port(self.requested_port)
            self._port_name = port_name
            with serial.Serial(
                port_name,
                self.baud,
                timeout=0.25,
                write_timeout=0.25,
            ) as device:
                device.reset_input_buffer()
                while not self._stop.is_set():
                    data = device.readline()
                    if not data:
                        continue
                    line = data.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    sample = self._parse_line(line)
                    with self._lock:
                        self._last_line = line
                        if sample is not None:
                            self._latest = sample
                            self._values.append(sample.distance_m)
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    @property
    def port_name(self) -> str:
        return self._port_name or self.requested_port


class CameraSource:
    def __init__(
        self,
        camera_id: int,
        width: int,
        height: int,
        fps: int,
        scan_max: int,
    ) -> None:
        self.requested_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.scan_max = scan_max
        self.camera_id = camera_id
        self.capture: Optional[cv2.VideoCapture] = None

    @staticmethod
    def _open(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def start(self) -> None:
        if self.requested_id >= 0:
            candidates = [self.requested_id]
        else:
            candidates = list(range(self.scan_max))

        opened: list[int] = []
        for index in candidates:
            capture = self._open(index, self.width, self.height, self.fps)
            ok, _ = capture.read() if capture.isOpened() else (False, None)
            capture.release()
            if ok:
                opened.append(index)

        if not opened:
            raise RuntimeError("No readable camera found")

        # External USB cameras normally receive the highest OpenCV index.
        self.camera_id = opened[-1]
        self.capture = self._open(
            self.camera_id,
            self.width,
            self.height,
            self.fps,
        )
        for _ in range(6):
            self.capture.read()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.capture is None:
            return False, None
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class GeometryV2Engine:
    def __init__(
        self,
        config_path: Path,
        weights_path: Path,
        device_name: str,
        score_threshold: float,
        dimensions_lhw: np.ndarray,
        camera_matrix: np.ndarray,
        fp16: bool,
    ) -> None:
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self.device = torch.device(device_name)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.dimensions_lhw = dimensions_lhw.astype(np.float32)
        self.camera_matrix = camera_matrix.astype(np.float32)

        cfg = smoke_cfg.clone()
        cfg.merge_from_file(str(config_path))
        cfg.merge_from_list(
            [
                "MODEL.DEVICE",
                str(self.device),
                "TEST.DETECTIONS_THRESHOLD",
                str(score_threshold),
                "TEST.FP16",
                "True" if self.fp16 else "False",
            ]
        )
        cfg.freeze()
        self.cfg = cfg
        self.transforms = build_transforms(cfg, is_train=False)
        self.model = build_detection_model(cfg).to(self.device).eval()

        checkpoint = torch.load(
            str(weights_path),
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def infer(
        self,
        frame_bgr: np.ndarray,
        camera_height_m: float,
    ) -> tuple[list[Detection], float]:
        model_view = center_crop_resize(frame_bgr, MODEL_WIDTH, MODEL_HEIGHT)
        rgb = cv2.cvtColor(model_view, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        size = np.array([MODEL_WIDTH, MODEL_HEIGHT], dtype=np.float32)
        center = size / 2.0
        output_size = [
            MODEL_WIDTH // self.cfg.MODEL.BACKBONE.DOWN_RATIO,
            MODEL_HEIGHT // self.cfg.MODEL.BACKBONE.DOWN_RATIO,
        ]
        trans_mat = get_transfrom_matrix([center, size], output_size)

        target = ParamsList(image_size=(MODEL_WIDTH, MODEL_HEIGHT), is_train=False)
        target.add_field("trans_mat", trans_mat.astype(np.float32))
        target.add_field("K", self.camera_matrix)
        target.add_field("h_cam", np.float32(camera_height_m))
        target.add_field("dimensions", self.dimensions_lhw)
        image_tensor, target = self.transforms(image, target)
        image_tensor = image_tensor.to(self.device)
        target = target.to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.fp16,
            ):
                output = self.model([image_tensor], [target])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        rows = output.detach().cpu().numpy()
        detections = [
            Detection(
                score=float(row[13]),
                alpha=float(row[1]),
                bbox=row[2:6].astype(np.float32),
                dimensions_hwl=row[6:9].astype(np.float32),
                location_xyz=row[9:12].astype(np.float32),
                rotation_y=float(row[12]),
            )
            for row in rows
        ]
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections, elapsed_ms


def center_crop_resize(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = width / height
    if source_aspect > target_aspect:
        crop_width = max(1, int(round(height * target_aspect)))
        left = max(0, (width - crop_width) // 2)
        cropped = frame[:, left : left + crop_width]
    else:
        crop_height = max(1, int(round(width / target_aspect)))
        top = max(0, (height - crop_height) // 2)
        cropped = frame[top : top + crop_height, :]
    return cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def box_corners_3d(detection: Detection) -> np.ndarray:
    height, width, length = detection.dimensions_hwl
    x = np.array(
        [length / 2, length / 2, -length / 2, -length / 2] * 2,
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 0, -height, -height, -height, -height], dtype=np.float32)
    z = np.array(
        [width / 2, -width / 2, -width / 2, width / 2] * 2,
        dtype=np.float32,
    )
    cosine = math.cos(detection.rotation_y)
    sine = math.sin(detection.rotation_y)
    rotation = np.array(
        [[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]],
        dtype=np.float32,
    )
    corners = rotation @ np.vstack([x, y, z])
    return corners + detection.location_xyz.reshape(3, 1)


def project_corners(corners: np.ndarray, camera_matrix: np.ndarray) -> Optional[np.ndarray]:
    if np.any(corners[2] <= 0.05):
        return None
    points = camera_matrix @ corners
    points = points[:2] / points[2:3]
    if not np.all(np.isfinite(points)):
        return None
    return points.T.astype(np.int32)


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    camera_matrix: np.ndarray,
) -> None:
    edge_pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    palette = ((46, 204, 113), (0, 196, 255), (255, 150, 64))
    for index, detection in enumerate(detections[:5]):
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
        x1 = int(np.clip(x1, 0, frame.shape[1] - 1))
        y1 = int(np.clip(y1, 0, frame.shape[0] - 1))
        x2 = int(np.clip(x2, 0, frame.shape[1] - 1))
        y2 = int(np.clip(y2, 0, frame.shape[0] - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        points = project_corners(box_corners_3d(detection), camera_matrix)
        if points is not None:
            for start, end in edge_pairs:
                cv2.line(
                    frame,
                    tuple(points[start]),
                    tuple(points[end]),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        label = (
            f"#{index + 1} {detection.score:.2f}  "
            f"x={detection.location_xyz[0]:+.2f}m "
            f"z={detection.location_xyz[2]:.2f}m"
        )
        draw_label(frame, label, (x1, max(22, y1 - 7)), color)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.48,
) -> None:
    (width, height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1,
    )
    x, y = origin
    cv2.rectangle(
        image,
        (x, y - height - baseline - 6),
        (x + width + 8, y + baseline + 2),
        (20, 23, 28),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def put_panel_text(
    panel: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = (225, 230, 238),
    scale: float = 0.52,
    thickness: int = 1,
) -> int:
    cv2.putText(
        panel,
        text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return y + int(30 * max(1.0, scale))


def render_dashboard(
    model_view: np.ndarray,
    detections: list[Detection],
    camera_matrix: np.ndarray,
    camera_id: int,
    laser_port: str,
    laser_baud: int,
    laser_sample: Optional[LaserSample],
    last_laser_line: str,
    laser_error: str,
    camera_height_m: Optional[float],
    frame_fps: float,
    inference_ms: float,
    device: torch.device,
    paused: bool,
) -> np.ndarray:
    annotated = model_view.copy()
    draw_detections(annotated, detections, camera_matrix)
    view = cv2.resize(annotated, (960, 720), interpolation=cv2.INTER_LINEAR)
    panel = np.full((720, 360, 3), (24, 28, 34), dtype=np.uint8)
    green = (76, 210, 141)
    amber = (68, 190, 245)
    red = (92, 92, 240)
    muted = (155, 165, 178)

    y = 42
    y = put_panel_text(panel, "GEOMETRY V2 LIVE", y, (246, 248, 250), 0.66, 2)
    cv2.line(panel, (22, y), (338, y), (58, 65, 75), 1)
    y += 34

    laser_color = green if laser_sample is not None else red
    y = put_panel_text(panel, "CAMERA HEIGHT", y, muted, 0.43)
    height_text = "--" if camera_height_m is None else f"{camera_height_m:.3f} m"
    y = put_panel_text(
        panel,
        height_text,
        y + 4,
        laser_color,
        1.08,
        2,
    )

    y += 28
    cv2.line(panel, (22, y), (338, y), (58, 65, 75), 1)
    y += 32
    y = put_panel_text(panel, "LASER", y, muted, 0.43)
    if laser_sample is not None:
        strength = "-" if laser_sample.strength is None else str(laser_sample.strength)
        y = put_panel_text(
            panel,
            f"{laser_sample.distance_m:.3f} m  strength {strength}",
            y + 2,
            green,
            0.54,
            1,
        )
    else:
        y = put_panel_text(panel, "waiting for valid range", y + 2, red, 0.50, 1)
    y = put_panel_text(panel, f"{laser_port} @ {laser_baud}", y, muted, 0.42)
    status_line = laser_error or last_laser_line or "no serial data"
    if len(status_line) > 42:
        status_line = status_line[:39] + "..."
    y = put_panel_text(panel, status_line, y, muted, 0.37)

    y += 15
    cv2.line(panel, (22, y), (338, y), (58, 65, 75), 1)
    y += 32
    y = put_panel_text(panel, "INFERENCE", y, muted, 0.43)
    y = put_panel_text(
        panel,
        f"{len(detections)} detections",
        y + 2,
        green if detections else amber,
        0.58,
        1,
    )
    y = put_panel_text(panel, f"{inference_ms:.1f} ms  |  {frame_fps:.1f} FPS", y, muted, 0.46)
    y = put_panel_text(panel, f"{device.type.upper()}  camera {camera_id}", y, muted, 0.43)

    y += 18
    for index, detection in enumerate(detections[:3]):
        yaw_deg = math.degrees(detection.rotation_y)
        y = put_panel_text(
            panel,
            f"#{index + 1} score {detection.score:.3f}",
            y,
            green,
            0.45,
            1,
        )
        y = put_panel_text(
            panel,
            (
                f"x {detection.location_xyz[0]:+.2f}  "
                f"y {detection.location_xyz[1]:.2f}  "
                f"z {detection.location_xyz[2]:.2f} m"
            ),
            y,
            muted,
            0.39,
        )
        y = put_panel_text(panel, f"yaw {yaw_deg:+.1f} deg", y, muted, 0.39)
        y += 6

    state = "PAUSED" if paused else "LIVE"
    state_color = amber if paused else green
    cv2.circle(panel, (28, 688), 6, state_color, -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        state,
        (43, 694),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        state_color,
        1,
        cv2.LINE_AA,
    )
    return np.hstack([view, panel])


def write_log_row(
    writer: Optional[csv.writer],
    timestamp: float,
    laser_sample: Optional[LaserSample],
    camera_height_m: Optional[float],
    detections: list[Detection],
    inference_ms: float,
) -> None:
    if writer is None:
        return
    best = detections[0] if detections else None
    writer.writerow(
        [
            f"{timestamp:.6f}",
            "" if laser_sample is None else f"{laser_sample.distance_m:.4f}",
            "" if camera_height_m is None else camera_height_m,
            len(detections),
            "" if best is None else best.score,
            "" if best is None else best.location_xyz[0],
            "" if best is None else best.location_xyz[1],
            "" if best is None else best.location_xyz[2],
            "" if best is None else best.rotation_y,
            inference_ms,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera and TFmini-S laser inference for geometry_v2",
    )
    parser.add_argument("--camera-id", type=int, default=-1, help="-1 scans and selects the highest readable index")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-scan-max", type=int, default=5)
    parser.add_argument("--laser-port", default="auto")
    parser.add_argument("--laser-baud", type=int, default=115200)
    parser.add_argument("--laser-channel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--laser-min-m", type=float, default=0.10)
    parser.add_argument("--laser-max-m", type=float, default=12.0)
    parser.add_argument("--laser-min-strength", type=int, default=1)
    parser.add_argument("--laser-stale-s", type=float, default=0.5)
    parser.add_argument("--laser-median-window", type=int, default=7)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fx", type=float, default=float(DEFAULT_K[0, 0]))
    parser.add_argument("--fy", type=float, default=float(DEFAULT_K[1, 1]))
    parser.add_argument("--cx", type=float, default=float(DEFAULT_K[0, 2]))
    parser.add_argument("--cy", type=float, default=float(DEFAULT_K[1, 2]))
    parser.add_argument("--length-m", type=float, default=float(DEFAULT_DIMS_LHW[0]))
    parser.add_argument("--height-m", type=float, default=float(DEFAULT_DIMS_LHW[1]))
    parser.add_argument("--width-m", type=float, default=float(DEFAULT_DIMS_LHW[2]))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames; 0 runs until quit")
    parser.add_argument("--save-last-frame", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera_matrix = np.array(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dimensions_lhw = np.array(
        [args.length_m, args.height_m, args.width_m],
        dtype=np.float32,
    )

    print(f"[model] loading {args.weights}")
    engine = GeometryV2Engine(
        config_path=args.config.resolve(),
        weights_path=args.weights.resolve(),
        device_name=args.device,
        score_threshold=args.score_threshold,
        dimensions_lhw=dimensions_lhw,
        camera_matrix=camera_matrix,
        fp16=args.fp16,
    )
    _, warmup_ms = engine.infer(
        np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8),
        1.0,
    )
    print(
        f"[model] ready on {engine.device}; fp16={engine.fp16}; "
        f"warmup={warmup_ms:.1f}ms"
    )

    laser = LaserReader(
        port=args.laser_port,
        baud=args.laser_baud,
        channel=args.laser_channel,
        min_distance_m=args.laser_min_m,
        max_distance_m=args.laser_max_m,
        min_strength=args.laser_min_strength,
        median_window=args.laser_median_window,
    )
    camera = CameraSource(
        camera_id=args.camera_id,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        scan_max=args.camera_scan_max,
    )

    log_handle = None
    log_writer = None
    if args.log_csv is not None:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_csv.open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_handle)
        log_writer.writerow(
            [
                "timestamp",
                "laser_distance_m",
                "camera_height_m",
                "detection_count",
                "best_score",
                "best_x_m",
                "best_y_m",
                "best_z_m",
                "best_rotation_y_rad",
                "inference_ms",
            ]
        )

    paused = False
    last_dashboard: Optional[np.ndarray] = None
    frame_times: deque[float] = deque(maxlen=30)
    frame_count = 0
    detections: list[Detection] = []
    inference_ms = 0.0

    try:
        laser.start()
        camera.start()
        print(f"[camera] index={camera.camera_id}")
        print(f"[laser] port={args.laser_port} baud={args.laser_baud}")
        print("[ui] ESC/q quit, SPACE pause, s screenshot, r reset laser filter")

        while True:
            ok, raw_frame = camera.read()
            if not ok or raw_frame is None:
                raise RuntimeError(f"Camera {camera.camera_id} stopped delivering frames")

            now = time.monotonic()
            frame_times.append(now)
            model_view = center_crop_resize(raw_frame, MODEL_WIDTH, MODEL_HEIGHT)
            laser_sample, laser_line, laser_error = laser.snapshot(args.laser_stale_s)
            if laser_sample is not None:
                camera_height_m: Optional[float] = laser_sample.distance_m
            else:
                camera_height_m = None
                detections = []
                inference_ms = 0.0

            if not paused and camera_height_m is not None:
                detections, inference_ms = engine.infer(raw_frame, camera_height_m)

            fps = 0.0
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            dashboard = render_dashboard(
                model_view=model_view,
                detections=detections,
                camera_matrix=camera_matrix,
                camera_id=camera.camera_id,
                laser_port=laser.port_name,
                laser_baud=args.laser_baud,
                laser_sample=laser_sample,
                last_laser_line=laser_line,
                laser_error=laser_error,
                camera_height_m=camera_height_m,
                frame_fps=fps,
                inference_ms=inference_ms,
                device=engine.device,
                paused=paused,
            )
            last_dashboard = dashboard
            write_log_row(
                log_writer,
                time.time(),
                laser_sample,
                camera_height_m,
                detections,
                inference_ms,
            )
            if log_handle is not None and frame_count % 30 == 0:
                log_handle.flush()

            if not args.no_display:
                cv2.imshow("Geometry V2 Live", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("r"):
                    laser.reset_filter()
                elif key == ord("s"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    output = BUNDLE_DIR / "captures" / f"live_{stamp}.png"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output), dashboard)
                    print(f"[capture] {output}")

            frame_count += 1
            if args.frames > 0 and frame_count >= args.frames:
                break
    finally:
        camera.close()
        laser.close()
        if log_handle is not None:
            log_handle.close()
        cv2.destroyAllWindows()

    if args.save_last_frame is not None and last_dashboard is not None:
        args.save_last_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_last_frame), last_dashboard)
        print(f"[output] {args.save_last_frame}")
    print(f"[done] frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
