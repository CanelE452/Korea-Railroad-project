# =============================================================================
# ⚠ DEPRECATED — 자율주행 신규 코드는 depth_cam/calib/can/ 사용 권장.
#   이 패키지의 lift/reach/fold 템플릿은 calib/can/protocol.py 로 흡수됨.
#   (canlib 하드 import 라 Kvaser SDK 없는 환경에선 import 자체가 실패 —
#    calib/can 은 mock fallback 이 있어 어디서나 동작)
#   autolifter/*.py 하드코딩 시퀀스 호환을 위해 유지.
# =============================================================================
from .controller import AutonomousForkliftController
from .types import OperationMode
from .exceptions import CANConnectionError

__all__ = ["AutonomousForkliftController", "OperationMode", "CANConnectionError"]
