# tools/bench_heartbeat.py
# =============================================================================
# 현장 브링업 1단계: 하트비트만 60초 송신.
#
# 목적:
#   - Kvaser 채널/비트레이트 정상 확인 (CanKing/candump 로 버스 에러 감시)
#   - CAN_ID_FAMILY 확정: 차량이 0x02E3 control 에 무반응이면
#       CAN_ID_FAMILY=E4 python bench_heartbeat.py   로 재시험
#
# 사용: python tools/bench_heartbeat.py [--sec 60]
# =============================================================================
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.control import can_init, can_close, is_mock
from calib.can.protocol import CAN_ID_FAMILY, HEARTBEAT_ID, CAN_CONTROL_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=60.0)
    args = ap.parse_args()

    print(f"[bench 1/4] 하트비트 {args.sec:.0f}s — "
          f"ID family=0x{CAN_ID_FAMILY:02X}, HB=0x{HEARTBEAT_ID:03X}, "
          f"CTRL=0x{CAN_CONTROL_ID:03X}")
    if not can_init():
        print("❌ CAN 초기화 실패")
        sys.exit(1)
    if is_mock():
        print("⚠ MOCK 모드 (canlib 없음) — 실차에서 다시 실행할 것")
    try:
        # can_init 이 TX 스레드(하트비트 200ms + stop 재송신) 기동 — 대기만
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.sec:
            remain = args.sec - (time.monotonic() - t0)
            print(f"  하트비트 송신 중... {remain:5.1f}s 남음", end="\r")
            time.sleep(1.0)
        print("\n✅ 완료 — CanKing/candump 에서 버스 에러 0건인지 확인")
        print("   프레임 로그: depth_cam/logs/can_*.jsonl")
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        can_close()


if __name__ == "__main__":
    main()
