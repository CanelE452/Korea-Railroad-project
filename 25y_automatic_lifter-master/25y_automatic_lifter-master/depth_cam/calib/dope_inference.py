# calib/dope_inference.py
# -----------------------------------------------------------------------------
# DOPE 6D pose inference adapter.
#
# - challengenight.pth (DopeNetwork, VGG-19 backbone, 9 belief + 16 affinity) 로드.
# - BGR image → 9 keypoint (image px, v4 camera-facing convention).
# - 추출된 9 keypoint 는 calib.pose6d_adapter.keypoints9_to_align_vars 로 전달되어
#   (ψ_pallet_deg, d_lateral_m, d_forward_m) 으로 변환된다.
#
# 구현 참고: challenge/scripts/run_live.py + run_live_io.py
#   1) 입력 이미지를 height=400 으로 resize (DOPE 학습 입력 size 와 호환).
#   2) ImageNet mean/std 로 normalize 후 forward.
#   3) 마지막 stage 의 belief map (vertex2: [9, 50, ~50]) 채널별 peak 추출.
#   4) peak 좌표 (50 grid) × 8 = 400-input 좌표 → ÷proc_scale = 원본 이미지 좌표.
#   5) channel peak val 이 threshold 미만이면 그 keypoint 는 invalid (-1, -1).
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from scipy.ndimage import gaussian_filter


# Deep_Object_Pose/common 을 sys.path 에 추가 (DopeNetwork / ModelData 임포트용)
_THIS_DIR = Path(__file__).resolve().parent
# depth_cam/calib/ → depth_cam/ → 25y_automatic_lifter-master/25y_automatic_lifter-master/
# → 25y_automatic_lifter-master/ → FoundationPose/
_REPO_ROOT = _THIS_DIR.parents[3]
_DOPE_COMMON = _REPO_ROOT / "Deep_Object_Pose" / "common"
if str(_DOPE_COMMON) not in sys.path:
    sys.path.insert(0, str(_DOPE_COMMON))

# detector.py 가 `from cuboid_pnp_solver import *` 로 pyrr (Quaternion) 까지
# transitively 끌어들이지만, 본 어댑터는 PnP 를 calib.pose6d_adapter 가 직접 수행하므로
# CuboidPNPSolver 가 필요 없다. pyrr 미설치 환경에서도 import 가 가능하도록
# 더미 모듈을 주입한다 (detector 가 실제 cuboid_pnp_solver 의 심볼을 사용하지 않음).
import types as _types
if "cuboid_pnp_solver" not in sys.modules:
    _stub = _types.ModuleType("cuboid_pnp_solver")
    _stub.__dict__["CuboidPNPSolver"] = None
    sys.modules["cuboid_pnp_solver"] = _stub
if "pyrr" not in sys.modules:
    _pyrr_stub = _types.ModuleType("pyrr")
    _pyrr_stub.__dict__["Quaternion"] = None
    _pyrr_stub.__dict__["matrix33"] = None
    sys.modules["pyrr"] = _pyrr_stub

# Deep_Object_Pose/common/detector.py 에서 정의된 클래스.
# ModelData 가 내부적으로 DopeNetwork 인스턴스 생성 + state_dict load + .cuda().eval() 한다.
from detector import ModelData  # type: ignore  # noqa: E402


_IMAGENET_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class DopePoseEstimator:
    """DOPE belief-map 기반 9 keypoint 추출기.

    Note:
        - VGG-19 (DopeNetwork) 가중치만 지원. Mobilenet 변형은 미지원.
        - challengenight.pth 는 학습 시 height=400 (run_live.py 와 동일 proc_scale)
          기준으로 동작. input_size 인자는 호환 위해 노출하지만 내부적으로 400 으로 고정.
    """

    def __init__(
        self,
        weights_path: str,
        input_height: int = 400,
        peak_threshold: float = 0.30,
        peak_sigma: float = 3.0,
        device: str = None,
    ):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DOPE weights not found: {weights_path}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_height = int(input_height)
        self.peak_threshold = float(peak_threshold)
        self.peak_sigma = float(peak_sigma)

        # DataParallel checkpoint 인지 자동 판별
        sd = torch.load(weights_path, map_location="cpu")
        first_key = next(iter(sd.keys())) if isinstance(sd, dict) and len(sd) > 0 else ""
        parallel = isinstance(first_key, str) and first_key.startswith("module.")

        print(f"[DopePose] weights : {weights_path}")
        print(f"[DopePose] device  : {self.device}")
        print(f"[DopePose] parallel ckpt: {parallel}")

        # ModelData.load_net_model() 는 내부에서 DopeNetwork().cuda() + state_dict load.
        # 본 코드베이스는 GPU 전제이지만 CPU fallback 가능하도록 wrapping.
        if self.device.startswith("cuda"):
            self.model = ModelData(name="pallet", net_path=weights_path, parallel=parallel)
            self.model.load_net_model()
            self.net = self.model.net  # already .cuda().eval()
        else:
            # CPU 추론용 — ModelData 의 .cuda() 우회.
            from detector import DopeNetwork  # type: ignore
            from collections import OrderedDict
            net = DopeNetwork()
            state_dict = sd
            if parallel:
                new_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_sd[k[7:]] = v
                state_dict = new_sd
            net.load_state_dict(state_dict)
            net.eval()
            self.net = net
            self.model = None

        # 마지막 추론의 proc_scale (디버깅/시각화용)
        self.last_proc_scale: float = 1.0

    # ------------------------------------------------------------------ forward
    def _forward(self, img_rgb_np: np.ndarray):
        """RGB(H,W,3,uint8) → (vertex2, aff)  — 마지막 stage tensor."""
        t = _IMAGENET_TF(img_rgb_np)
        if self.device.startswith("cuda"):
            t = Variable(t).cuda().unsqueeze(0)
        else:
            t = Variable(t).unsqueeze(0)
        with torch.no_grad():
            out, seg = self.net(t)
        return out[-1][0], seg[-1][0]

    def _extract_peaks(self, vertex2) -> List[dict]:
        """채널별 Gaussian smoothed peak. scale_factor=8 (output 50 → input 400)."""
        peaks = []
        for ch in range(vertex2.size(0)):
            raw = vertex2[ch].cpu().numpy()
            sm = gaussian_filter(raw, sigma=self.peak_sigma)
            val = float(sm.max())
            idx = np.unravel_index(sm.argmax(), sm.shape)
            peaks.append({"val": val, "x": int(idx[1]) * 8, "y": int(idx[0]) * 8})
        return peaks

    # ------------------------------------------------------------------ public
    def infer_keypoints9(
        self,
        bgr: np.ndarray,
    ) -> Optional[List[Tuple[float, float]]]:
        """BGR image → 9 keypoint (image px, 원본 해상도).

        Returns:
            [(u, v)] x 9 — channel 0~7 = cuboid corner (v4 convention),
                          channel 8 = centroid.
                          peak confidence 가 peak_threshold 미만이면 (NaN, NaN).
            검출이 전혀 안 되면 (모든 채널 invalid) None.
        """
        if bgr is None or bgr.ndim != 3:
            return None
        h, w = bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        # height=input_height 으로 resize (run_live.py 와 동일 방식)
        proc_scale = float(self.input_height) / float(h)
        new_w = int(round(w * proc_scale)) & ~7   # 8의 배수로 정렬 (VGG stride 호환)
        new_w = max(new_w, 8)
        img_small = cv2.resize(bgr, (new_w, self.input_height))
        img_rgb = img_small[..., ::-1].copy()  # BGR → RGB
        self.last_proc_scale = proc_scale

        vertex2, _aff = self._forward(img_rgb)
        peaks = self._extract_peaks(vertex2)

        # peak (400-input coord) → 원본 coord (÷proc_scale).
        kps: List[Tuple[float, float]] = []
        valid_count = 0
        for pk in peaks:
            if pk["val"] >= self.peak_threshold:
                u = pk["x"] / proc_scale
                v = pk["y"] / proc_scale
                kps.append((float(u), float(v)))
                valid_count += 1
            else:
                kps.append((float("nan"), float("nan")))

        if valid_count < 4:
            # PnP 최소 요구치 미달
            return None
        return kps
