# calib/can — CAN 통신 통합 패키지 (2026-07 정리)
#
# 계층:
#   protocol.py — ID/템플릿/상수 단일 진실 (canlib 미의존, 순수 데이터)
#   mockcan.py  — canlib mock (DLL 없는 환경/테스트)
#   bus.py      — 전송 + 신뢰성 (TX 스레드/재송신/워치독/재초기화/프레임 로그)
#   commands.py — 고수준 명령 (주행 = 구 control.py 와 바이트 동일, 리프트 신규)
#
# 기존 코드 호환은 calib/control.py (shim) 를 통해 유지된다.
from .commands import *  # noqa: F401,F403
from .bus import BUS, is_mock  # noqa: F401
from . import protocol  # noqa: F401
