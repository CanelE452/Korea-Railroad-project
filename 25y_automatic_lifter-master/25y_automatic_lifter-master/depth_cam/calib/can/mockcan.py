# calib/can/mockcan.py
# =============================================================================
# Kvaser canlib mock — DLL 없는 개발/CI 환경에서 전체 파이프라인 실행용.
# (구 calib/control.py 의 mock 을 추출 + 테스트용 기록 채널 추가)
#
# 사용:
#   from calib.can.mockcan import load_canlib
#   canlib, Frame, real_ok = load_canlib()
# =============================================================================
from __future__ import annotations

import time


class MockCanlibError(Exception):
    pass


class MockChannel:
    def setBusOutputControl(self, *a, **k):
        pass

    def setBusParams(self, *a, **k):
        pass

    def busOn(self):
        pass

    def busOff(self):
        pass

    def close(self):
        pass

    def write(self, frame):
        pass

    def read(self, *a, **k):
        raise MockCanlibError("mock: no frame")


class RecordingChannel(MockChannel):
    """write 된 Frame 을 (ts, id, data, flags) 로 기록 — 테스트/골든 캡처용."""

    def __init__(self):
        self.records = []

    def write(self, frame):
        self.records.append({
            "ts": time.monotonic(),
            "id": int(frame.id),
            "data": [int(b) for b in frame.data],
            "flags": int(getattr(frame, "flags", 0)),
        })

    def clear(self):
        self.records.clear()

    def frames(self):
        """(id, data) 튜플 목록 — ts 무시 비교용."""
        return [(r["id"], tuple(r["data"])) for r in self.records]


class FailingChannel(MockChannel):
    """write 가 항상 실패 — bus 에러 복구 경로 테스트용."""

    def __init__(self, exc=None):
        self.exc = exc or MockCanlibError("mock: tx fail")
        self.attempts = 0

    def write(self, frame):
        self.attempts += 1
        raise self.exc


class _MockBitrate:
    BITRATE_1M = -1
    BITRATE_500K = -2
    BITRATE_250K = -3
    BITRATE_125K = -4
    BITRATE_100K = -5
    BITRATE_62K = -6
    BITRATE_50K = -7


class MockCanlib:
    canERR_NOTFOUND = -3
    canOPEN_ACCEPT_VIRTUAL = 0
    canBITRATE_500K = -2
    Bitrate = _MockBitrate
    Driver = type("Driver", (), {"NORMAL": 4})()
    canERR_NOMSG = MockCanlibError

    @staticmethod
    def openChannel(channel, flags=0):
        return MockChannel()

    @staticmethod
    def getNumberOfChannels():
        return 0


class MockFrame:
    def __init__(self, id_=0, data=b"", dlc=0, flags=0):
        self.id = id_
        self.data = data
        self.dlc = dlc
        self.flags = flags


def load_canlib():
    """실제 canlib 로드 시도, 실패 시 mock 반환.

    Returns:
        (canlib_module, Frame_class, real_ok: bool)
    """
    try:
        from canlib import canlib as real_canlib, Frame as RealFrame  # type: ignore
        return real_canlib, RealFrame, True
    except (ImportError, FileNotFoundError, OSError) as e:
        print(f"[CAN] canlib DLL load failed ({type(e).__name__}: {e}) — using mock canlib.")
        return MockCanlib(), MockFrame, False
