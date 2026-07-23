# calib/truck/lasers.py
# =============================================================================
# TFmini-S 듀얼 레이저 (좌/우 포크) 리더 + 판정 로직.
#
# 배선 미확정 (사용자 확인: 레이저 2개, 단일 시리얼인지 포트 2개인지 모름)
# → 두 모드 모두 지원, config.LASER_WIRING 으로 선택:
#   "single_port": 한 시리얼 스트림에 "L1 70cm strength=NNNN" / "L2 ..." 두 채널
#                  (truck_loading/live_geometry_v2.py 의 LaserReader 파서 확장)
#   "dual_port"  : COM 포트 2개 각각 1채널
#
# 설계 원칙: I/O(리더) 와 판정 로직(감지기) 분리.
#   EdgeDropDetector / ReleaseDetector 는 순수 로직 — 합성 트레이스로
#   완전 단위테스트 가능 (다이어그램 T15~T20 / T21~TD6).
# =============================================================================
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# 샘플/파서 (live_geometry_v2.py 의 LaserReader 파서와 동일 규격)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LaserSample:
    distance_m: float
    strength: Optional[int]
    timestamp: float
    raw: str


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


def parse_laser_line(line: str) -> list:
    """한 줄 파싱 → [(channel or None, distance_m, strength), ...].

    TFmini 채널 형식이면 채널 명시, named/bare 형식이면 channel=None
    (dual_port 모드에서 포트가 채널을 결정).
    """
    out = []
    matched = False
    for m in _TFMINI_PATTERN.finditer(line):
        matched = True
        out.append((
            int(m.group("channel")),
            float(m.group("distance")) / 100.0,
            int(m.group("strength")) if m.group("strength") else None,
        ))
    if matched:
        return out

    m = _NAMED_PATTERN.search(line)
    if m:
        d = float(m.group("distance"))
        unit = m.group("unit").lower()
        if unit == "mm":
            d /= 1000.0
        elif unit == "cm":
            d /= 100.0
        return [(None, d, None)]

    m = _BARE_PATTERN.match(line)
    if m:
        # bare 값은 관례상 cm
        return [(None, float(m.group("distance")) / 100.0, None)]
    return []


class _ChannelFilter:
    """채널별 유효성 검사 + median 필터."""

    def __init__(self, min_m: float, max_m: float, min_strength: int, window: int):
        self.min_m = min_m
        self.max_m = max_m
        self.min_strength = min_strength
        self.values: deque = deque(maxlen=max(1, window))
        self.latest: Optional[LaserSample] = None

    def push(self, distance_m: float, strength: Optional[int], raw: str) -> bool:
        if not (self.min_m <= distance_m <= self.max_m):
            return False
        if strength is not None and strength < self.min_strength:
            return False
        self.latest = LaserSample(distance_m, strength, time.monotonic(), raw)
        self.values.append(distance_m)
        return True

    def snapshot(self, stale_after_s: float) -> Optional[LaserSample]:
        s = self.latest
        if s is None or (time.monotonic() - s.timestamp) > stale_after_s:
            return None
        vals = sorted(self.values)
        med = vals[len(vals) // 2] if vals else s.distance_m
        return LaserSample(med, s.strength, s.timestamp, s.raw)

    def reset(self):
        self.values.clear()


class DualLaserReader:
    """좌/우 TFmini-S 리더 — 배선 모드 겸용.

    Args:
        wiring   : "single_port" | "dual_port"
        port     : single_port 모드의 포트 ("auto" 가능)
        port_l/r : dual_port 모드의 좌/우 포트
        ch_l/r   : single_port 모드에서 좌/우 채널 번호 (L1/L2)
    """

    def __init__(self, wiring: str = "single_port",
                 port: str = "auto", port_l: str = "auto", port_r: str = "auto",
                 baud: int = 115200, ch_l: int = 1, ch_r: int = 2,
                 min_m: float = 0.02, max_m: float = 12.0,
                 min_strength: int = 1, median_window: int = 5):
        self.wiring = wiring
        self.port = port
        self.port_l = port_l
        self.port_r = port_r
        self.baud = baud
        self.ch_l = ch_l
        self.ch_r = ch_r
        self._filt = {
            ch_l: _ChannelFilter(min_m, max_m, min_strength, median_window),
            ch_r: _ChannelFilter(min_m, max_m, min_strength, median_window),
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._last_line = ""
        self._error = ""

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def _discover_port(requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        from serial.tools import list_ports  # lazy import
        ports = list(list_ports.comports())
        if not ports:
            raise RuntimeError("No serial ports found")
        preferred = [
            p for p in ports
            if any(tok in f"{p.description} {p.manufacturer} {p.hwid}".lower()
                   for tok in ("ch340", "1a86:7523", "usb-serial", "cp210"))
        ]
        return (preferred or ports)[0].device

    def _serial_loop(self, port: str, fixed_channel: Optional[int]):
        """시리얼 읽기 루프. fixed_channel: dual_port 모드에서 이 포트의 채널."""
        import serial  # lazy import
        try:
            name = self._discover_port(port)
            with serial.Serial(name, self.baud, timeout=0.25) as dev:
                dev.reset_input_buffer()
                while not self._stop.is_set():
                    data = dev.readline()
                    if not data:
                        continue
                    line = data.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    with self._lock:
                        self._last_line = line
                        for ch, dist, strength in parse_laser_line(line):
                            ch = ch if ch is not None else fixed_channel
                            f = self._filt.get(ch)
                            if f is not None:
                                f.push(dist, strength, line)
        except Exception as e:
            with self._lock:
                self._error = f"{type(e).__name__}: {e}"

    def start(self):
        if self.wiring == "dual_port":
            targets = [(self.port_l, self.ch_l), (self.port_r, self.ch_r)]
        else:
            targets = [(self.port, None)]
        for port, fixed in targets:
            t = threading.Thread(target=self._serial_loop, args=(port, fixed),
                                 name=f"laser-{port}", daemon=True)
            t.start()
            self._threads.append(t)

    def close(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    # -------------------------------------------------------------- snapshot
    def snapshot(self, stale_after_s: float = 0.5
                 ) -> Tuple[Optional[LaserSample], Optional[LaserSample], str, str]:
        """(sample_L, sample_R, last_line, error). stale 채널은 None."""
        with self._lock:
            sl = self._filt[self.ch_l].snapshot(stale_after_s)
            sr = self._filt[self.ch_r].snapshot(stale_after_s)
            return sl, sr, self._last_line, self._error

    def reset_filters(self):
        with self._lock:
            for f in self._filt.values():
                f.reset()

    # 테스트/시뮬레이션 훅 — 시리얼 없이 샘플 주입
    def inject_sample(self, channel: int, distance_m: float,
                      strength: Optional[int] = None, raw: str = "inject"):
        with self._lock:
            f = self._filt.get(channel)
            if f is not None:
                f.push(distance_m, strength, raw)


# =============================================================================
# 순수 로직 감지기 (I/O 없음 — 합성 트레이스로 단위테스트)
# =============================================================================
class EdgeDropDetector:
    """적재면 모서리 감지 — 다이어그램 T15~T20/TD5.

    포크가 트럭 적재면 위로 진입하는 순간, 포크 후단 하부의 좌/우 레이저가
    (바닥까지 거리 → 적재면까지 거리로) '동시에 급감'한다.

    규칙:
      - drop = baseline - current 가 drop_thresh_m 초과 → 해당 채널 "급감"
      - 급감이 아니면 baseline 을 현재값으로 갱신 (T19 — 서행 변화 추종)
      - 좌/우 급감 시각 차가 sync_window_s 이내 → 후보
      - 후보가 confirm_n 회 연속 → "EDGE" (디바운스)
      - 어느 채널이든 값 없음(stale) → "FAULT" (FSM 은 STOP)
    """

    NONE = "NONE"
    EDGE = "EDGE"
    FAULT = "FAULT"

    def __init__(self, drop_thresh_m: float, sync_window_s: float, confirm_n: int):
        self.drop_thresh_m = float(drop_thresh_m)
        self.sync_window_s = float(sync_window_s)
        self.confirm_n = int(confirm_n)
        self.reset()

    def reset(self):
        self._baseline = {"L": None, "R": None}
        self._drop_ts = {"L": None, "R": None}
        self._confirm = 0

    def _update_channel(self, key: str, dist: float, now: float):
        base = self._baseline[key]
        if base is None:
            self._baseline[key] = dist
            return
        drop = base - dist
        if drop > self.drop_thresh_m:
            # 급감 — baseline 유지 (T19 의 "아니오" 분기만 갱신)
            if self._drop_ts[key] is None:
                self._drop_ts[key] = now
        else:
            self._baseline[key] = dist
            self._drop_ts[key] = None

    def update(self, now: float,
               dist_l: Optional[float], dist_r: Optional[float]) -> str:
        if dist_l is None or dist_r is None:
            self._confirm = 0
            return self.FAULT

        self._update_channel("L", float(dist_l), now)
        self._update_channel("R", float(dist_r), now)

        tl, tr = self._drop_ts["L"], self._drop_ts["R"]
        both = (tl is not None and tr is not None
                and abs(tl - tr) <= self.sync_window_s)
        if both:
            self._confirm += 1
            if self._confirm >= self.confirm_n:
                return self.EDGE
        else:
            self._confirm = 0
        return self.NONE


class ReleaseDetector:
    """팔레트 안착 판정 — 다이어그램 TD6.

    하강 중 laser_L < threshold AND laser_R < threshold 가
    confirm_n 회 연속이면 안착 (True). 값 없음(stale)은 카운터 리셋.
    """

    def __init__(self, threshold_m: float, confirm_n: int):
        self.threshold_m = float(threshold_m)
        self.confirm_n = int(confirm_n)
        self._confirm = 0

    def reset(self):
        self._confirm = 0

    def update(self, dist_l: Optional[float], dist_r: Optional[float]) -> bool:
        if dist_l is None or dist_r is None:
            self._confirm = 0
            return False
        if dist_l < self.threshold_m and dist_r < self.threshold_m:
            self._confirm += 1
        else:
            self._confirm = 0
        return self._confirm >= self.confirm_n
