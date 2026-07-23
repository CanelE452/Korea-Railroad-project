# calib/can/bus.py
# =============================================================================
# CAN 전송 계층 + 신뢰성 계층.
#
# 파렛트 단계 오류의 구조적 원인(fire-once 송신, 모드전환 settle 없음,
# except:pass, 프레임 로그 없음)을 해소하는 최소 장치:
#
#   1. 단일 TX 스레드가 채널을 독점 — 주기 재송신(하트비트/movement/control)
#   2. movement 재송신: 마지막 movement 프레임을 50ms 주기 재송신
#      (프레임 1개 유실이 오동작으로 이어지던 fire-once 제거)
#   3. TX 예외 → 에러 카운터 → 연속 N회 실패 시 busOff/busOn 재초기화(백오프)
#      → FSM 은 bus_healthy() 로 확인 (False 면 STOP 유지 권장)
#   4. 소프트웨어 워치독: keepalive() 가 WATCHDOG_S 이상 끊기면 정지 프레임만 송신
#      (FSM 프로세스 hang 시 차량 폭주 방지 — arm_watchdog() 호출 후부터 활성)
#   5. 전 프레임 JSONL 로그 (사후 분석 + 골든 테스트 기반)
#
# 테스트 훅:
#   inject_test_channel(ch) — TX 스레드 없이 채널만 교체 (골든 캡처용)
#   reset_sync_for_test(seed)
# =============================================================================
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from . import protocol as P
from .mockcan import load_canlib

canlib, Frame, _CANLIB_REAL = load_canlib()

# Mock send 로그 toggle (기존 CAN_MOCK_VERBOSE 유지)
_MOCK_VERBOSE = os.environ.get("CAN_MOCK_VERBOSE", "0") == "1"

# ---- 신뢰성 파라미터 ----
MOV_RESEND_PERIOD = 0.050    # movement 재송신 주기 (s)
CTRL_RESEND_PERIOD = 0.200   # control 재송신 주기 (s)
TX_TICK = 0.010              # TX 스레드 tick (s)
TX_ERR_REINIT_THRESHOLD = 5  # 연속 TX 실패 → 재초기화
REINIT_BACKOFF_S = [0.5, 1.0, 2.0, 4.0]   # 재초기화 백오프
WATCHDOG_S = 1.0             # keepalive 끊김 허용 시간 (s)


def is_mock() -> bool:
    """canlib DLL 없이 mock 으로 동작 중인지."""
    return not _CANLIB_REAL


class CanBus:
    """단일 채널 소유 + TX 스레드 + 신뢰성 계층."""

    def __init__(self):
        self._ch = None
        self._lock = threading.Lock()
        self._sync = P.SYNC_COUNTER_SEED
        self._extended = P.USE_EXTENDED_IDS

        # TX 스레드 상태
        self._tx_thread: Optional[threading.Thread] = None
        self._tx_stop = threading.Event()
        self._active_movement: str = "stop"
        self._active_control: str = "driving_mode"

        # 신뢰성 상태
        self._tx_err_count = 0
        self._reinit_count = 0
        self._healthy = True
        self._last_keepalive = 0.0
        self._watchdog_armed = False

        # 재초기화용 접속 파라미터
        self._channel_no = P.CAN_CHANNEL
        self._bitrate = P.CAN_BITRATE

        # 프레임 로그
        self._log_fh = None
        self._log_count = 0

    # ------------------------------------------------------------- 내부 유틸
    def _flags(self) -> int:
        return P.MessageFlag.EXT if self._extended else P.MessageFlag.STD

    def _next_sync(self) -> int:
        self._sync = P.next_sync(self._sync)
        return self._sync

    def _mk_control(self, ctrl_type: str) -> Frame:
        data = P.CONTROL_TEMPLATES.get(ctrl_type, P.CONTROL_TEMPLATES["driving_mode"]).copy()
        if ctrl_type != "emergency":
            data[4] = self._next_sync()
        return Frame(id_=P.CAN_CONTROL_ID, data=data, flags=self._flags())

    def _mk_movement(self, name: str) -> Frame:
        data = P.MOVEMENT_TEMPLATES.get(name, P.MOVEMENT_TEMPLATES["stop"])
        return Frame(id_=P.CAN_MOVEMENT_ID, data=data, flags=self._flags())

    def _mk_heartbeat(self) -> Frame:
        return Frame(id_=P.HEARTBEAT_ID, data=P.HEARTBEAT_DATA, flags=self._flags())

    def _log_frame(self, frame: Frame, src: str):
        if self._log_fh is None:
            return
        try:
            self._log_fh.write(json.dumps({
                "ts": round(time.time(), 4),
                "id": f"0x{int(frame.id):03X}",
                "data": [int(b) for b in frame.data],
                "src": src,
            }, ensure_ascii=False) + "\n")
            self._log_count += 1
            if self._log_count % 100 == 0:
                self._log_fh.flush()
        except Exception:
            pass  # 로그 실패가 제어를 막으면 안 됨

    def _write(self, frame: Frame, src: str = "") -> bool:
        """프레임 1개 송신. 성공 여부 반환 (에러 카운팅 포함)."""
        self._log_frame(frame, src)
        if self._ch is None:
            if _MOCK_VERBOSE:
                print(f"[MOCK SEND] id=0x{int(frame.id):03X}, "
                      f"data={[hex(b) for b in frame.data]}, src={src}")
            return True
        try:
            with self._lock:
                self._ch.write(frame)
            self._tx_err_count = 0
            self._healthy = True
            return True
        except Exception as e:
            self._tx_err_count += 1
            if self._tx_err_count == 1 or self._tx_err_count % 10 == 0:
                print(f"[CAN TX ERROR] ({self._tx_err_count}회 연속) {type(e).__name__}: {e}")
            if self._tx_err_count >= TX_ERR_REINIT_THRESHOLD:
                self._healthy = False
            return False

    def _try_reinit(self):
        """버스 재초기화 (백오프 적용). TX 스레드에서만 호출."""
        backoff = REINIT_BACKOFF_S[min(self._reinit_count, len(REINIT_BACKOFF_S) - 1)]
        self._reinit_count += 1
        print(f"[CAN] 버스 재초기화 시도 #{self._reinit_count} (backoff {backoff}s)")
        time.sleep(backoff)
        try:
            with self._lock:
                if self._ch is not None:
                    try:
                        self._ch.busOff()
                        self._ch.close()
                    except Exception:
                        pass
                self._ch = self._open_channel()
            self._tx_err_count = 0
            self._healthy = True
            print("[CAN] 버스 재초기화 성공")
        except Exception as e:
            print(f"[CAN] 버스 재초기화 실패: {e}")

    def _open_channel(self):
        ch = canlib.openChannel(self._channel_no)
        br_map = {
            1_000_000: canlib.Bitrate.BITRATE_1M,
            500_000: canlib.Bitrate.BITRATE_500K,
            250_000: canlib.Bitrate.BITRATE_250K,
            125_000: canlib.Bitrate.BITRATE_125K,
        }
        ch.setBusParams(br_map.get(self._bitrate, canlib.Bitrate.BITRATE_500K))
        ch.busOn()
        return ch

    # --------------------------------------------------------------- TX 루프
    def _tx_loop(self):
        last_hb = 0.0
        last_mov = 0.0
        last_ctrl = 0.0
        while not self._tx_stop.is_set():
            now = time.monotonic()

            # 워치독: keepalive 끊김 → 정지 프레임만
            wd_tripped = (
                self._watchdog_armed
                and (now - self._last_keepalive) > WATCHDOG_S
            )
            mov_name = "stop" if wd_tripped else self._active_movement

            if now - last_hb >= P.HEARTBEAT_PERIOD:
                self._write(self._mk_heartbeat(), src="tx:hb")
                last_hb = now

            if now - last_mov >= MOV_RESEND_PERIOD:
                self._write(self._mk_movement(mov_name),
                            src="tx:wd_stop" if wd_tripped else f"tx:mov:{mov_name}")
                last_mov = now

            if now - last_ctrl >= CTRL_RESEND_PERIOD:
                self._write(self._mk_control(self._active_control),
                            src=f"tx:ctrl:{self._active_control}")
                last_ctrl = now

            if not self._healthy:
                self._try_reinit()

            self._tx_stop.wait(TX_TICK)

    # --------------------------------------------------------------- 공개 API
    def init(self, channel: int = P.CAN_CHANNEL, bitrate: int = P.CAN_BITRATE,
             is_extended_id: bool = P.USE_EXTENDED_IDS,
             frame_log_dir: Optional[str] = None) -> bool:
        """채널 오픈 + 시작 안정화 버스트 + TX 스레드 기동. 성공 시 True."""
        self._channel_no = channel
        self._bitrate = bitrate
        self._extended = bool(is_extended_id)
        try:
            self._ch = self._open_channel()
        except Exception as e:
            print(f"[CAN INIT ERROR] {e}")
            self._ch = None
            return False

        # 프레임 로그 (기본: depth_cam/logs/can_*.jsonl)
        try:
            log_dir = Path(frame_log_dir) if frame_log_dir else (
                Path(__file__).resolve().parents[2] / "logs"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._log_fh = open(log_dir / f"can_{ts}.jsonl", "a", encoding="utf-8")
            print(f"[CAN] 프레임 로그: {log_dir / f'can_{ts}.jsonl'}")
        except Exception as e:
            print(f"[CAN] 프레임 로그 비활성 ({e})")
            self._log_fh = None

        # 시작 안정화: driving_mode + stop + heartbeat 5회 (기존 유지)
        for _ in range(5):
            self._write(self._mk_control("driving_mode"), src="init")
            self._write(self._mk_movement("stop"), src="init")
            self._write(self._mk_heartbeat(), src="init")
            time.sleep(0.005)

        self.start_tx_thread()
        return True

    def close(self):
        """정지 송신 후 스레드/버스 해제."""
        self.stop_tx_thread()
        self._write(self._mk_movement("stop"), src="close")
        if self._ch is not None:
            try:
                with self._lock:
                    self._ch.busOff()
                    self._ch.close()
            except Exception:
                pass
            self._ch = None
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        print("[CAN CLOSED]")

    def start_tx_thread(self):
        if self._tx_thread is not None and self._tx_thread.is_alive():
            return
        self._tx_stop.clear()
        self._tx_thread = threading.Thread(target=self._tx_loop, name="CANTx", daemon=True)
        self._tx_thread.start()

    def stop_tx_thread(self):
        self._tx_stop.set()
        t = self._tx_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._tx_thread = None

    # ---- 즉시 송신 (issue_command_* 가 사용 — 기존과 바이트/순서 동일) ----
    def send_control(self, name: str, src: str = "cmd") -> bool:
        return self._write(self._mk_control(name), src=src)

    def send_movement(self, name: str, src: str = "cmd") -> bool:
        self._active_movement = name  # 재송신 루프에 반영
        return self._write(self._mk_movement(name), src=src)

    def send_heartbeat(self, src: str = "cmd") -> bool:
        return self._write(self._mk_heartbeat(), src=src)

    def set_active_control(self, name: str):
        """control 재송신 루프가 유지할 control 프레임 지정 (즉시 송신 없음)."""
        self._active_control = name

    # ---- 신뢰성 상태 ----
    def bus_healthy(self) -> bool:
        return self._healthy

    def keepalive(self):
        self._last_keepalive = time.monotonic()

    def arm_watchdog(self):
        """워치독 활성화 — FSM 루프 시작 직전 호출. 이후 keepalive() 주기 호출 필요."""
        self._last_keepalive = time.monotonic()
        self._watchdog_armed = True

    def disarm_watchdog(self):
        self._watchdog_armed = False

    # ---- 테스트 훅 ----
    def inject_test_channel(self, ch):
        """TX 스레드 없이 채널만 교체 (골든 캡처/단위 테스트 전용)."""
        self.stop_tx_thread()
        self._ch = ch

    def reset_sync_for_test(self, seed: int = P.SYNC_COUNTER_SEED):
        self._sync = seed


# 모듈 싱글턴 — 기존 control.py 의 모듈-함수 스타일 유지
BUS = CanBus()
