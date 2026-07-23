# truck_main.py
# =============================================================================
# Phase B (트럭 적재) 실행 엔트리.
#
# 두 가지 사용법:
#   1) 단독 실행 (현장 테스트 — 파렛트 단계 재실행 불필요):
#        python truck_main.py [--camera-id N] [--no-display] [--dry-run]
#   2) main_rec.py 에서 파렛트 완료 후 핸드오프:
#        from truck_main import run_phase_b
#        run_phase_b(skip_can_init=True)   # CAN 은 이미 초기화됨
#
# 구성 (다이어그램 T0):
#   Camera2 (포크 전면부) + SMOKE geometry_v2  → 트럭 6D pose (T2)
#   TFmini-S Laser L/R (포크 후단 하부)         → 모서리/안착 감지 + 카메라 높이
#   RealSense IMU (RelYawReader)               → 회전 종료 판정 (rel_yaw)
#
# --dry-run: SMOKE/카메라 없이 레이저+FSM 만 동작 (배선/레이저 점검용).
# =============================================================================
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# depth_cam/ 을 import root 로 (calib.* 접근)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import calib.config as cfg
from calib.control import (
    can_init, can_close, is_mock, keepalive, arm_watchdog, bus_healthy,
)
from calib.fsm.commands import CommandExecutor
from calib.fsm.truck import TruckMachine
from calib.truck.lasers import DualLaserReader
from calib.truck.truck_adapter import TruckStateGate


def _build_laser_reader() -> DualLaserReader:
    return DualLaserReader(
        wiring=cfg.LASER_WIRING,
        port=cfg.LASER_PORT, port_l=cfg.LASER_PORT_L, port_r=cfg.LASER_PORT_R,
        baud=cfg.LASER_BAUD, ch_l=cfg.LASER_CH_L, ch_r=cfg.LASER_CH_R,
    )


def _build_rel_yaw_reader():
    """RealSense IMU rel_yaw 리더 (eval/imu_yaw.py 재사용). 실패 시 None."""
    try:
        repo_inner = _THIS_DIR.parent   # 25y_automatic_lifter-master(inner)
        if str(repo_inner) not in sys.path:
            sys.path.insert(0, str(repo_inner))
        from eval.imu_yaw import RelYawReader
        r = RelYawReader()
        r.start()
        return r
    except Exception as e:
        print(f"⚠ RelYawReader 초기화 실패 ({e}) — rel_yaw 없이 진행 (회전 상태 진입 시 대기)")
        return None


def run_phase_b(skip_can_init: bool = False, camera_id: int = -1,
                dry_run: bool = False, display: bool = False,
                max_runtime_s: float = 0.0) -> str:
    """Phase B 루프 실행. 종료 시 최종 상태 문자열 반환 ("DONE"/"FAULT"/...).

    Args:
        skip_can_init : main_rec.py 핸드오프 시 True (CAN 이미 초기화)
        camera_id     : Camera2 OpenCV 인덱스 (-1 = 스캔)
        dry_run       : SMOKE/카메라 없이 레이저+FSM 만
        max_runtime_s : 0 이면 무제한, >0 이면 안전 상한
    """
    # ---- CAN ----
    if not skip_can_init:
        ok = can_init()
        if ok and is_mock():
            print("⚠ CAN MOCK 모드 — 실제 송수신 없음 (FSM 로직 확인용)")
        elif not ok:
            print("❌ CAN 초기화 실패 — MOCK 진행")
    arm_watchdog()

    # ---- 센서 (T0: SWITCH_SENSOR_INPUT) ----
    lasers = _build_laser_reader()
    lasers.start()

    perception = None
    if not dry_run:
        from calib.truck.smoke_source import TruckPerception
        perception = TruckPerception(
            bundle_dir=cfg.TRUCK_SMOKE_BUNDLE_DIR,
            camera_id=camera_id,
            score_thr=cfg.TRUCK_DET_SCORE_THR,
        )
        perception.start()
        print("✅ SMOKE 트럭 인지 초기화 완료")

    rel_yaw_reader = _build_rel_yaw_reader()

    gate = TruckStateGate(
        score_thr=cfg.TRUCK_DET_SCORE_THR,
        confirm_n=cfg.TRUCK_CONFIRM_FRAMES,
        cam_to_fork_t=cfg.CAM2_TO_FORK_T,
        cam_to_fork_rpy_deg=cfg.CAM2_TO_FORK_RPY_DEG,
    )
    execu = CommandExecutor()
    machine = TruckMachine(execu=execu)

    print("▶ Phase B 시작 (Ctrl+C 로 중단 = 즉시 STOP)")
    t0 = time.monotonic()
    last_print = 0.0
    try:
        while True:
            keepalive()   # 워치독 갱신 — 이 루프가 죽으면 bus 가 STOP 강등

            if not bus_healthy():
                print("⚠ CAN bus unhealthy — 재초기화 대기 (STOP 유지)")
                time.sleep(0.5)
                continue

            # ---- 레이저 스냅샷 ----
            sl, sr, _line, err = lasers.snapshot(stale_after_s=cfg.LASER_STALE_S)
            laser_l = sl.distance_m if sl else None
            laser_r = sr.distance_m if sr else None

            # ---- 트럭 pose (TRUCK_CHECK 에서만 소비되지만 게이트는 항상 갱신) ----
            truck_state = None
            if perception is not None and machine.state in ("WAIT_SENSORS", "TRUCK_CHECK"):
                # camera_height_m = 레이저 거리 (번들 규약: 유효값 없으면 추론 대기)
                h = laser_l if laser_l is not None else laser_r
                if h is not None:
                    det = perception.infer_best(camera_height_m=h)
                    truck_state = gate.update(det)

            rel_yaw = None
            if rel_yaw_reader is not None and rel_yaw_reader.available():
                rel_yaw = rel_yaw_reader.rel_yaw

            # ---- FSM 1 tick ----
            lines = machine.step(
                truck_state=truck_state, rel_yaw=rel_yaw,
                laser_l=laser_l, laser_r=laser_r,
            )

            now = time.monotonic()
            if now - last_print > 0.5 and lines:
                for text, _color in lines:
                    print(f"  {text}")
                if err:
                    print(f"  [laser err] {err}")
                last_print = now

            if machine.state in ("DONE", "FAULT"):
                print(f"■ Phase B 종료: {machine.state}"
                      + (f" ({machine.fault_reason})" if machine.state == "FAULT" else ""))
                break
            if max_runtime_s > 0 and (now - t0) > max_runtime_s:
                print("■ Phase B 최대 실행시간 초과 — STOP")
                execu.exec("STOP")
                break

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n■ 사용자 중단 — STOP")
        execu.exec("STOP")
    finally:
        try:
            lasers.close()
        except Exception:
            pass
        try:
            if perception is not None:
                perception.close()
        except Exception:
            pass
        if not skip_can_init:
            can_close()
    return machine.state


def main():
    ap = argparse.ArgumentParser(description="Phase B (트럭 적재) 단독 실행")
    ap.add_argument("--camera-id", type=int, default=-1,
                    help="Camera2 OpenCV 인덱스 (-1 = 스캔)")
    ap.add_argument("--dry-run", action="store_true",
                    help="SMOKE/카메라 없이 레이저+FSM 만 (배선 점검)")
    ap.add_argument("--max-runtime", type=float, default=0.0,
                    help="안전 상한 (s), 0=무제한")
    args = ap.parse_args()
    state = run_phase_b(
        skip_can_init=False, camera_id=args.camera_id,
        dry_run=args.dry_run, max_runtime_s=args.max_runtime,
    )
    sys.exit(0 if state == "DONE" else 1)


if __name__ == "__main__":
    main()
