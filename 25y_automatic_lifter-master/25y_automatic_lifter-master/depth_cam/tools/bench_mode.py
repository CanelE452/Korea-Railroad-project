# tools/bench_mode.py
# =============================================================================
# 현장 브링업 2단계: 모드 전환 왕복 (driving ↔ lift ↔ reach ↔ folding).
#
# 목적: 차량 모드 표시등/거동으로 모드 프레임 수신 확인.
#       settle(MODE_SWITCH_SETTLE_S) 적용 상태에서 전환 안정성 관찰.
#
# 사용: python tools/bench_mode.py [--cycles 3] [--hold 2.0]
# =============================================================================
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.control import can_init, can_close, is_mock
from calib.can.bus import BUS
from calib.can.commands import MODE_FRAME_REPEAT, MODE_SWITCH_SETTLE_S

MODES = ["driving_mode", "lift_mode", "reach_mode", "folding_mode"]


def switch_mode(name: str):
    BUS.send_movement("stop", src="bench:mode")
    for _ in range(MODE_FRAME_REPEAT):
        BUS.send_control(name, src="bench:mode")
        BUS.send_heartbeat(src="bench:mode")
        time.sleep(0.005)
    BUS.set_active_control(name)
    time.sleep(MODE_SWITCH_SETTLE_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--hold", type=float, default=2.0, help="모드당 유지 시간(s)")
    args = ap.parse_args()

    print(f"[bench 2/4] 모드 전환 {args.cycles}회 왕복 (settle={MODE_SWITCH_SETTLE_S}s)")
    if not can_init():
        sys.exit(1)
    if is_mock():
        print("⚠ MOCK 모드 — 실차에서 다시 실행할 것")
    try:
        for c in range(args.cycles):
            for m in MODES:
                print(f"  [{c + 1}/{args.cycles}] → {m}  (차량 모드 표시 확인)")
                switch_mode(m)
                time.sleep(args.hold)
        switch_mode("driving_mode")
        print("✅ 완료 — 각 전환마다 차량이 해당 모드로 진입했는지 육안 확인")
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        can_close()


if __name__ == "__main__":
    main()
