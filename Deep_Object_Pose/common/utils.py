"""
NVIDIA from jtremblay@gmail.com (NVlabs DOPE fork).

이 파일은 분리된 utils_* 모듈의 backwards-compat re-export shim.
기존 `from utils import ...` import path 가 모두 그대로 작동.

분리된 모듈:
  utils_math.py     vector / angle 헬퍼 (length, py_ang, ...)
  utils_loaders.py  파일 탐색 (loadimages, loadweights, ...)
  utils_belief.py   belief / affinity map 생성 + 시각화
  utils_dataset.py  CleanVisiiDopeLoader + augmentation 클래스
  utils_viz.py      make_grid / Draw / save_image
"""
# Re-export — 기존 코드 호환 유지
from utils_math import (
    length, dot_product, normalize, determinant, inner_angle, py_ang,
)
from utils_loaders import (
    default_loader, append_dot, loadimages, loadweights, loadimages_inference,
)
from utils_belief import (
    VisualizeAffinityMap, VisualizeBeliefMap, GenerateMapAffinity,
    getAfinityCenter, CreateBeliefMap,
)
from utils_dataset import (
    CleanVisiiDopeLoader, crop,
    AddRandomContrast, AddRandomBrightness, AddNoise,
)
from utils_viz import make_grid, save_image, Draw, get_image_grid
