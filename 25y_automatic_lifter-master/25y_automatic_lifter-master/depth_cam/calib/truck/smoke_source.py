# calib/truck/smoke_source.py
# =============================================================================
# truck_loading/ SMOKE(geometry_v2) 번들 lazy 래퍼.
#
# 번들(truck_loading/geometry_v2_live_camera_laser_ready_20260723/)은 그대로
# 두고 sys.path 로 import 한다 — 코드 복제 없음. torch/cv2 는 이 모듈의
# 메서드 안에서만 import 되므로, 어댑터/감지기 테스트는 torch 없이 동작.
#
# 다이어그램 매핑: T0(센서 전환) → T1(캡처) → T2(TRUCK_POSE_INFERENCE).
# camera_height_m 는 레이저 L1 거리 (번들 규약 유지 — 임의값 대입 금지).
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple


class TruckPerception:
    """Camera2 + SMOKE 추론 소스.

    Args:
        bundle_dir : geometry_v2 번들 경로 (config.TRUCK_SMOKE_BUNDLE_DIR)
        camera_id  : OpenCV 카메라 인덱스 (-1 = 스캔)
        intrinsics : (fx, fy, cx, cy) — None 이면 번들 기본값
        dims_lhw   : 트럭 3D prior (length, height, width) — None 이면 번들 기본값
        score_thr  : detection score threshold
        device     : "auto" | "cuda" | "cpu"
    """

    def __init__(self, bundle_dir: str, camera_id: int = -1,
                 intrinsics: Optional[Tuple[float, float, float, float]] = None,
                 dims_lhw: Optional[Tuple[float, float, float]] = None,
                 score_thr: float = 0.25, device: str = "auto", fp16: bool = False):
        self.bundle_dir = Path(bundle_dir)
        self.camera_id = camera_id
        self.intrinsics = intrinsics
        self.dims_lhw = dims_lhw
        self.score_thr = score_thr
        self.device = device
        self.fp16 = fp16

        self._engine = None
        self._camera = None
        self._lg = None   # live_geometry_v2 모듈

    # ------------------------------------------------------------------ lazy
    def _import_bundle(self):
        if self._lg is not None:
            return self._lg
        if not self.bundle_dir.is_dir():
            raise FileNotFoundError(f"SMOKE 번들 없음: {self.bundle_dir}")
        if str(self.bundle_dir) not in sys.path:
            sys.path.insert(0, str(self.bundle_dir))
        import live_geometry_v2 as lg  # noqa: E402 — 번들 모듈
        self._lg = lg
        return lg

    def start(self):
        """모델 로드 + 카메라 오픈 (torch/cv2 필요)."""
        import numpy as np
        lg = self._import_bundle()

        K = lg.DEFAULT_K.copy()
        if self.intrinsics is not None:
            fx, fy, cx, cy = self.intrinsics
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dims = (np.array(self.dims_lhw, dtype=np.float32)
                if self.dims_lhw is not None else lg.DEFAULT_DIMS_LHW)

        self._engine = lg.GeometryV2Engine(
            config_path=lg.DEFAULT_CONFIG,
            weights_path=lg.DEFAULT_WEIGHTS,
            device_name=self.device,
            score_threshold=self.score_thr,
            dimensions_lhw=dims,
            camera_matrix=K,
            fp16=self.fp16,
        )
        self._camera = lg.CameraSource(
            camera_id=self.camera_id, width=1920, height=1080, fps=30, scan_max=5,
        )
        self._camera.start()

    def close(self):
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    # ------------------------------------------------------------------ 추론
    def infer_best(self, camera_height_m: float):
        """1 프레임 캡처 + 추론 → 최고 score Detection tuple 또는 None.

        Returns:
            None 또는 (score, location_xyz, rotation_y, dimensions_hwl)
            — TruckStateGate.update() 입력 형식.
        """
        if self._engine is None or self._camera is None:
            raise RuntimeError("start() 먼저 호출")
        ok, frame = self._camera.read()
        if not ok or frame is None:
            return None
        detections, _ms = self._engine.infer(frame, float(camera_height_m))
        if not detections:
            return None
        best = detections[0]   # score 내림차순 정렬됨
        return (
            float(best.score),
            [float(v) for v in best.location_xyz],
            float(best.rotation_y),
            [float(v) for v in best.dimensions_hwl],
        )
