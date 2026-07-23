# tests_can/conftest.py
# depth_cam/ 을 import root 로 추가 (calib.* 패키지 import 용).
# tests/eval 의 conftest 패턴과 동일.
import os
import sys

_DEPTH_CAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DEPTH_CAM_DIR not in sys.path:
    sys.path.insert(0, _DEPTH_CAM_DIR)
