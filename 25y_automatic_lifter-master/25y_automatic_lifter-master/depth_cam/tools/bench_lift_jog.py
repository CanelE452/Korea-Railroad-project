# tools/bench_lift_jog.py
# =============================================================================
# 현장 브링업 3단계: 리프트 조그 — ⚠ lift_up/lift_down 코드 모순 해소 단계.
#
# 배경: v1(control_forklift.py)+forklift_ctrl 은 up=0x25/down=0x15,
#       v2(control_forklift_v2.py)는 반대. 현재 기본값은 다수결(up=0x25).
#
# 이 스크립트는 "LIFT_UP 코드"를 짧게 펄스한다.
#   → 포크가 **올라가면**: 기본값 맞음. 아무것도 할 필요 없음.
#   → 포크가 **내려가면**: 반대. 환경변수로 뒤집어 재확인:
#        LIFT_CODES_SWAPPED=1 python tools/bench_lift_jog.py
#     맞으면 calib/can/protocol.py 의 기본값을 영구 수정하거나
#     실행 환경에 LIFT_CODES_SWAPPED=1 을 고정할 것.
#
# 사용: python tools/bench_lift_jog.py [--pulse 0.3] [--updown]
#   --updown: up 펄스 후 down 펄스도 실행 (양방향 확인)
# =============================================================================
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.control import (
    can_init, can_close, is_mock,
    issue_lift_up, issue_lift_down, issue_lift_stop,
)
from calib.can.protocol import LIFT_UP_CODE, LIFT_DOWN_CODE


def pulse(action_fn, name: str, sec: float):
    print(f"  {name} 펄스 {sec:.1f}s ...")
    action_fn()          # 모드 시퀀스 포함 (정지→lift_mode→settle→동작)
    time.sleep(sec)
    issue_lift_stop()    # 해제 + 주행 모드 복귀
    time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulse", type=float, default=0.3, help="펄스 시간(s)")
    ap.add_argument("--updown", action="store_true", help="down 펄스도 실행")
    args = ap.parse_args()

    print(f"[bench 3/4] 리프트 조그 — LIFT_UP=0x{LIFT_UP_CODE:02X} "
          f"LIFT_DOWN=0x{LIFT_DOWN_CODE:02X}")
    print("  ⚠ 포크 주변 사람/장애물 없는지 확인. E-stop 대기.")
    input("  준비되면 Enter...")
    if not can_init():
        sys.exit(1)
    if is_mock():
        print("⚠ MOCK 모드 — 실차에서 다시 실행할 것")
    try:
        pulse(issue_lift_up, "LIFT_UP", args.pulse)
        print("  → 포크가 올라갔으면 기본값 OK / 내려갔으면 LIFT_CODES_SWAPPED=1 로 재시험")
        if args.updown:
            pulse(issue_lift_down, "LIFT_DOWN", args.pulse)
            print("  → 포크가 내려갔으면 OK")
        print("✅ 완료")
    except KeyboardInterrupt:
        print("\n중단 — 리프트 정지")
        issue_lift_stop()
    finally:
        can_close()


if __name__ == "__main__":
    main()
