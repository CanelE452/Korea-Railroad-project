# calib/perception.py
# DOPE 6D pose 추론 (2026-05-22 통합).
#
# 기존 YOLO seg 구현은 perception_yolo_legacy.py 로 보존.
#
# 외부 인터페이스:
#     Perception(model_path=None).infer(color_img_bgr, depth_frame=None, K=...)
#       → {
#           "ok":            bool,
#           "R_pallet":      (3,3) np.float64  (default_z180 적용 완료)
#           "t_pallet_cm":   (3,)  np.float64  (centroid, cm)
#           "raw_points":    list of 9 of Optional[(x,y)]   (전처리된 이미지 좌표)
#           "proj_points":   list of 9 of Optional[(x,y)]   (PnP 결과 reprojection)
#           "K_proc":        (3,3) np.float64   전처리 스케일 적용된 K (시각화용)
#           "reason":        str
#           "confirmed":     bool (temporal confirm 통과 시 True)
#         }
#
# Twin-PnP 검증 결과로 다음 두 가지 고정 (depth_cam/tools/twin_pnp_check.py):
#   1) PALLET_PNP_CONTRACT_Z180 = True → cuboid vertices @ diag([-1,-1,+1])
#   2) PALLET_(WIDTH/DEPTH/HEIGHT)_M = (1.0, 1.2, 0.15)

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch

from .config import (
    MODEL_PATH,
    DOPE_BELIEF_THRESHOLD, DOPE_BELIEF_THRESH_MAP, DOPE_BELIEF_THRESH_POINTS,
    DOPE_BELIEF_THRESH_ANGLE, DOPE_BELIEF_SIGMA, DOPE_INPUT_HEIGHT,
    DOPE_GATE_MIN_KP, DOPE_GATE_MAX_REPROJ_PX,
    DOPE_GATE_Z_MIN_M, DOPE_GATE_Z_MAX_M, DOPE_GATE_DEPTH_PNP_REL,
    DOPE_TEMPORAL_CONFIRM_FRAMES,
    PALLET_WIDTH_M, PALLET_DEPTH_M, PALLET_HEIGHT_M,
    PALLET_PNP_CONTRACT_Z180,
)

# ── DOPE 모듈 import (FoundationPose/Deep_Object_Pose/common) ─────────────────
_DEPTH_CAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_DEPTH_CAM_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Deep_Object_Pose", "common"))
from cuboid import Cuboid3d  # noqa: E402
from cuboid_pnp_solver import CuboidPNPSolver  # noqa: E402
from detector import ModelData, ObjectDetector  # noqa: E402
from pyrr import Quaternion, matrix33  # noqa: E402

# ── run_live 분리 모듈 재사용 (challenge/scripts) ─────────────────────────────
sys.path.insert(0, os.path.join(_REPO_ROOT, "challenge", "scripts"))
from run_live_io import run_forward, scale_K  # noqa: E402
from run_live_gates import evaluate_result  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────

class _DopeCfg:
    """ObjectDetector.find_object_poses 가 기대하는 cfg 인터페이스."""
    mask_edges = 1
    mask_faces = 1
    vertex = 1
    softmax = 1000

    def __init__(self):
        self.threshold = DOPE_BELIEF_THRESHOLD
        self.thresh_map = DOPE_BELIEF_THRESH_MAP
        self.thresh_points = DOPE_BELIEF_THRESH_POINTS
        self.thresh_angle = DOPE_BELIEF_THRESH_ANGLE
        self.sigma = DOPE_BELIEF_SIGMA


def _make_pnp_solver(pallet_width_m: float, pallet_height_m: float, pallet_depth_m: float):
    """default Cuboid3d 기반 PnP solver. dim 은 cm 단위로 변환."""
    dim_cm = [pallet_width_m * 100.0, pallet_height_m * 100.0, pallet_depth_m * 100.0]
    return CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(dim_cm))


class Perception:
    """DOPE 기반 6D pose 추론기. main_rec.py 의 기존 Perception 클래스를 대체."""

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.weights = model_path or MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("==========[Perception / DOPE 6D]==========")
        print(f"[Perception] weights : {self.weights}")
        print(f"[Perception] device  : {self.device}")
        print(f"[Perception] dim (W,H,L) m : "
              f"({PALLET_WIDTH_M}, {PALLET_HEIGHT_M}, {PALLET_DEPTH_M})")
        print(f"[Perception] PnP Z180 flip: {PALLET_PNP_CONTRACT_Z180}")
        print("==========================================")

        if not os.path.isfile(self.weights):
            raise FileNotFoundError(f"DOPE weight not found: {self.weights}")

        sd_keys = list(torch.load(self.weights, map_location="cpu").keys())
        parallel = sd_keys[0].startswith("module.") if sd_keys else False
        print(f"[Perception] DataParallel checkpoint: {parallel}")

        self.model = ModelData(name="pallet", net_path=self.weights, parallel=parallel)
        self.model.load_net_model()

        # PnP solver
        self.pnp_solver = _make_pnp_solver(PALLET_WIDTH_M, PALLET_HEIGHT_M, PALLET_DEPTH_M)
        self.pnp_solver.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))

        # twin-PnP 검증 결과 적용: default Cuboid3d.vertices @ diag([-1,-1,+1]) (Z180)
        if PALLET_PNP_CONTRACT_Z180:
            base = np.array(self.pnp_solver._cuboid3d.get_vertices(), dtype=np.float64)
            flipped = base @ np.diag([-1.0, -1.0, 1.0])
            # Cuboid3d 인스턴스의 내부 _vertices 캐시 덮어쓰기
            self.pnp_solver._cuboid3d._vertices = flipped  # type: ignore
            print("[Perception] PnP contract overridden: default @ diag([-1,-1,+1])")

        # 추론 cfg
        self.cfg = _DopeCfg()

        # temporal confirm 카운터
        self._consecutive_ok: int = 0
        self._confirm_threshold: int = DOPE_TEMPORAL_CONFIRM_FRAMES

    # ── main interface ────────────────────────────────────────────────────────
    def infer(self, color_img_bgr: np.ndarray,
              depth_frame=None,
              K: Optional[np.ndarray] = None) -> dict:
        """RGB(+ optional depth)에서 6D pose 추론. 자세한 반환 형식은 모듈 docstring 참조."""
        if K is None:
            raise ValueError("Perception.infer() requires K (camera intrinsic).")

        h, w = color_img_bgr.shape[:2]

        # 전처리: run_live.py 와 동일 — h=400 으로 resize, K 도 같은 스케일
        proc_scale = float(DOPE_INPUT_HEIGHT) / h
        new_w = int(w * proc_scale) & ~7  # 8 배수
        img_small = cv2.resize(color_img_bgr, (new_w, DOPE_INPUT_HEIGHT))
        K_small = scale_K(np.asarray(K, dtype=np.float64), proc_scale)
        self.pnp_solver.set_camera_intrinsic_matrix(K_small)

        # forward
        img_rgb = img_small[..., ::-1].copy()
        vertex2, aff = run_forward(self.model.net, img_rgb)

        # find_object_poses
        try:
            results = ObjectDetector.find_object_poses(vertex2, aff, self.pnp_solver, self.cfg)
        except Exception as e:
            return self._fail_result(K_small, proc_scale, f"find_object_poses exception: {e!s}")

        if not results:
            self._consecutive_ok = 0
            return self._fail_result(K_small, proc_scale, "no_results")

        # sanity gate
        gates_cfg = {
            "min_detected_keypoints": DOPE_GATE_MIN_KP,
            "max_reproj_error_px": DOPE_GATE_MAX_REPROJ_PX,
            "z_min_m": DOPE_GATE_Z_MIN_M,
            "z_max_m": DOPE_GATE_Z_MAX_M,
            "depth_pnp_z_max_rel": DOPE_GATE_DEPTH_PNP_REL,
            "cuboid_edge_ratio_tol": 0.30,
        }

        best = None
        last_reason = "no_valid_result"
        for result in results:
            depth_cm = None
            raw = result.get("raw_points")
            if depth_frame is not None and raw is not None and raw[8] is not None:
                u = raw[8][0] / proc_scale
                v = raw[8][1] / proc_scale
                d = self._sample_depth(depth_frame, u, v)
                if d is not None:
                    depth_cm = d * 100.0
            ok, reason, _ = evaluate_result(result, gates_cfg, depth_cm, K_small)
            last_reason = reason
            if ok:
                best = result
                break

        if best is None:
            self._consecutive_ok = 0
            return self._fail_result(K_small, proc_scale, last_reason)

        # 성공
        loc = np.array(best["location"], dtype=np.float64)        # cm
        quat_xyzw = best["quaternion"]
        q = Quaternion(quat_xyzw)
        R_pallet = np.array(matrix33.create_from_quaternion(q), dtype=np.float64)

        self._consecutive_ok += 1
        confirmed = (self._consecutive_ok >= self._confirm_threshold)

        return {
            "ok":            True,
            "R_pallet":      R_pallet,
            "t_pallet_cm":   loc,
            "raw_points":    best.get("raw_points"),
            "proj_points":   best.get("projected_points"),
            "K_proc":        K_small,
            "proc_scale":    proc_scale,
            "reason":        "ok",
            "confirmed":     confirmed,
        }

    # ── 내부 ──────────────────────────────────────────────────────────────────
    def _fail_result(self, K_proc, proc_scale, reason: str) -> dict:
        return {
            "ok":           False,
            "R_pallet":     None,
            "t_pallet_cm":  None,
            "raw_points":   None,
            "proj_points":  None,
            "K_proc":       K_proc,
            "proc_scale":   proc_scale,
            "reason":       reason,
            "confirmed":    False,
        }

    @staticmethod
    def _sample_depth(depth_frame, x: float, y: float, radius: int = 3) -> Optional[float]:
        if depth_frame is None:
            return None
        fw = depth_frame.get_width()
        fh = depth_frame.get_height()
        vals = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = int(x) + dx, int(y) + dy
                if 0 <= nx < fw and 0 <= ny < fh:
                    d = depth_frame.get_distance(nx, ny)
                    if d > 0.05:
                        vals.append(d)
        return float(np.median(vals)) if vals else None

    def infer_front(self, color_img):
        """YOLO 시그니처 호환. 6D 통합 후에는 호출하지 말 것."""
        raise NotImplementedError("DOPE Perception: use infer(color_img, depth_frame, K) instead.")
