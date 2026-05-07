"""Blender 합성 데이터 설정값."""

import os
import numpy as np

try:
    import bpy
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(bpy.data.filepath), "..", "..", ".."))
    if not os.path.isdir(os.path.join(PROJECT_ROOT, "data", "pallet")):
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
except Exception:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Render
NUM_FRAMES = 30
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FX = FY = 615.111
CX = 320.0
CY = 240.0

# Paths
def _next_output_dir():
    """자동으로 다음 버전 폴더 생성: test_blender_v1, v2, v3..."""
    base = os.path.join(PROJECT_ROOT, "data", "pallet")
    v = 1
    while os.path.exists(os.path.join(base, f"test_blender_v{v}")):
        v += 1
    return os.path.join(base, f"test_blender_v{v}")

OUTPUT_DIR = _next_output_dir()
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlay")

# Scene object names
PALLET_NAMES = ["Pallet_1", "Pallet_2", "Pallet_3"]  # Pallet_0 has no mesh in Blender

DISTRACTOR_NAMES = [
    "Barrel_01", "Barrel_1", "concrete_road_barrier",
    "Sketchfab_model", "Sketchfab_model.001", "Sketchfab_model.002",
    "TrafficCone_1", "TrafficCone_2",
]

BOX_NAMES = [f"PalletBox_{i}" for i in range(7)]

# ORIENTATION_OVERRIDES (XYZ intrinsic Euler degrees) -- DO NOT MODIFY
ORIENTATION_OVERRIDES = {
    "Pallet_0": (180, 0, 90),
    "Pallet_1": (90, 0, 0),
    "Pallet_2": (90, 0, 0),
    "Pallet_3": (0, 0, 0),
}

# Canonical pallet dimensions (Y=UP convention)
CANONICAL_BBOX_MIN = np.array([0.0, 0.0, 0.0])
CANONICAL_BBOX_MAX = np.array([1.1, 0.15, 1.1])
# Actual measured pallet top Z for each model (from Blender mesh bounding box)
PALLET_TOP_Z = {
    "Pallet_1": 0.137,
    "Pallet_2": 0.218,
    "Pallet_3": 0.145,
}
PALLET_SURFACE_Z = 0.15  # fallback default

K = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1],
], dtype=np.float64)

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CORNER_COLORS_RGB = [
    (255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0),
    (0, 0, 255), (0, 128, 255), (128, 0, 255), (255, 0, 255),
]

HDRI_BASE_STRENGTH = 1.0
