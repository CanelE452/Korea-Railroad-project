# tools/bench_drive_jog.py
# =============================================================================
# 현장 브링업 4단계: 주행 조그 — 전/후진/회전/저속전진 짧은 펄스.
#
# 목적:
#   - 새 CAN 코어(재송신 루프)에서 주행 프레임 정상 동작 확인
#   - 재송신으로 인한 '중복/과다 반응' 없는지 관찰
#   - forward_slow (트럭 모서리 탐색용 신규 템플릿) 실측 속도 확인
#     → 실측값으로 config.SLOW_FWD_MPS 갱신
#
# 사용: python tools/bench_drive_jog.py [--pulse 0.5]
# =============================================================================
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calib.control import (
    can_init, can_close, is_mock,
    issue_command_forward, issue_command_forward_slow, issue_command_backward,
    issue_command_rotate_in_place, issue_command_stop,
)

JOGS = [
    ("전진", issue_command_forward),
    ("후진", issue_command_backward),
    ("저속 전진 (forward_slow)", issue_command_forward_slow),
    ("제자리 좌회전", lambda: issue_command_rotate_in_place(+1)),
    ("제자리 우회전", lambda: issue_command_rotate_in_place(-1)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulse", type=float, default=0.5, help="펄스 시간(s)")
    args = ap.parse_args()

    print(f"[bench 4/4] 주행 조그 — 펄스 {args.pulse:.1f}s")
    print("  ⚠ 전후좌우 2m 이상 확보. E-stop 대기.")
    input("  준비되면 Enter...")
    if not can_init():
        sys.exit(1)
    if is_mock():
        print("⚠ MOCK 모드 — 실차에서 다시 실행할 것")
    try:
        for name, fn in JOGS:
            input(f"  다음: {name} — Enter 로 실행...")
            fn()
            time.sleep(args.pulse)
            issue_command_stop()
            time.sleep(1.0)
        print("✅ 완료 — 각 동작이 1회씩만, 부드럽게 수행됐는지 확인")
        print("   forward_slow 이동거리/시간으로 config.SLOW_FWD_MPS 갱신할 것")
    except KeyboardInterrupt:
        print("\n중단 — 정지")
        issue_command_stop()
    finally:
        can_close()


if __name__ == "__main__":
    main()
