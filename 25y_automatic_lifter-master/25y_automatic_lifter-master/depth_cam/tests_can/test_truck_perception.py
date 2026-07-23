# tests_can/test_truck_perception.py
# =============================================================================
# calib/truck 순수 로직 테스트 — 하드웨어/torch 불필요.
#   - truck_adapter: 합성 Detection → (ψ, d_lat, d_fwd) 수학 검증
#   - EdgeDropDetector: 동시 급감 / 한쪽만 / 시간차 / stale / 디바운스
#   - ReleaseDetector: 해제 임계 디바운스
#   - parse_laser_line: L1/L2 채널 파싱
# =============================================================================
from __future__ import annotations

import math

import pytest

from calib.truck.truck_adapter import truck_detection_to_align_vars, TruckStateGate
from calib.truck.lasers import (
    EdgeDropDetector, ReleaseDetector, parse_laser_line, DualLaserReader,
)

# 표준 트럭 prior: (h, w, l) = (2.0, 1.75, 5.15) — 번들 DEFAULT_DIMS_LHW 의 hwl 순서
DIMS_HWL = (2.0, 1.75, 5.15)


# =============================================================================
# truck_adapter
# =============================================================================
def test_adapter_facing_side_center():
    """트럭 옆면을 정면(법선이 카메라 향함)으로 5m 앞에서 본 경우.

    rotation_y=0 이면 박스 z축(폭 방향) = 카메라 z축 → 가까운 옆면은
    z = 5.0 - w/2, 법선은 -Z (카메라 쪽) → ψ=0, d_lat=0.
    """
    psi, d_lat, d_fwd = truck_detection_to_align_vars(
        location_xyz=[0.0, 1.0, 5.0], rotation_y=0.0, dimensions_hwl=DIMS_HWL,
    )
    assert psi == pytest.approx(0.0, abs=1e-6)
    assert d_lat == pytest.approx(0.0, abs=1e-6)
    assert d_fwd == pytest.approx(5.0 - DIMS_HWL[1] / 2.0, abs=1e-6)


def test_adapter_lateral_offset_sign():
    """트럭이 오른쪽(+X)에 있으면 d_lateral 은 음수 (부호 반전 규약)."""
    psi, d_lat, d_fwd = truck_detection_to_align_vars(
        location_xyz=[1.5, 1.0, 5.0], rotation_y=0.0, dimensions_hwl=DIMS_HWL,
    )
    assert d_lat == pytest.approx(-1.5, abs=1e-6)


def test_adapter_yawed_truck():
    """트럭이 +20° 돌아 있으면 ψ 도 ±20° (부호는 wrap 규약 내 일관성만 확인)."""
    psi, _, _ = truck_detection_to_align_vars(
        location_xyz=[0.0, 1.0, 5.0], rotation_y=math.radians(20.0),
        dimensions_hwl=DIMS_HWL,
    )
    assert abs(psi) == pytest.approx(20.0, abs=1e-4)

    psi_neg, _, _ = truck_detection_to_align_vars(
        location_xyz=[0.0, 1.0, 5.0], rotation_y=math.radians(-20.0),
        dimensions_hwl=DIMS_HWL,
    )
    assert psi_neg == pytest.approx(-psi, abs=1e-4), "yaw 부호 대칭이어야 함"


def test_adapter_picks_near_side():
    """멀리 있는 옆면이 아니라 가까운 옆면을 골라야 함 — d_fwd < 중심거리."""
    _, _, d_fwd = truck_detection_to_align_vars(
        location_xyz=[0.0, 1.0, 6.0], rotation_y=0.0, dimensions_hwl=DIMS_HWL,
    )
    assert d_fwd < 6.0


def test_adapter_extrinsic_identity_default():
    """extrinsic 기본값(항등) — 명시 적용과 무적용 결과 동일 (회귀 없음)."""
    a = truck_detection_to_align_vars([0.5, 1.0, 4.0], 0.3, DIMS_HWL)
    b = truck_detection_to_align_vars([0.5, 1.0, 4.0], 0.3, DIMS_HWL,
                                      cam_to_fork_t=[0, 0, 0],
                                      cam_to_fork_rpy_deg=[0, 0, 0])
    assert a == pytest.approx(b)


def test_state_gate_confirm():
    gate = TruckStateGate(score_thr=0.25, confirm_n=3)
    det = (0.9, [0.0, 1.0, 5.0], 0.0, DIMS_HWL)
    assert gate.update(det) is None          # 1
    assert gate.update(det) is None          # 2
    state = gate.update(det)                  # 3 — 확정
    assert state is not None
    psi, d_lat, d_fwd = state
    assert psi == pytest.approx(0.0, abs=1e-6)

    # 저신뢰 검출 → 카운터 리셋
    assert gate.update((0.1, [0, 1, 5], 0.0, DIMS_HWL)) is None
    assert gate.update(det) is None          # 다시 1부터


# =============================================================================
# EdgeDropDetector (다이어그램 T15~T20/TD5)
# =============================================================================
def _mk_edge():
    return EdgeDropDetector(drop_thresh_m=0.30, sync_window_s=0.3, confirm_n=2)


def test_edge_simultaneous_drop():
    d = _mk_edge()
    t = 0.0
    # 바닥 주행 (1.5m) — baseline 형성
    for _ in range(5):
        assert d.update(t, 1.5, 1.5) == d.NONE
        t += 0.1
    # 양쪽 동시 급감 (적재면 위 진입: 1.5 → 0.4)
    assert d.update(t, 0.4, 0.4) == d.NONE   # confirm 1/2
    t += 0.1
    assert d.update(t, 0.4, 0.4) == d.EDGE   # confirm 2/2 → 확정


def test_edge_single_side_no_fire():
    d = _mk_edge()
    t = 0.0
    for _ in range(5):
        d.update(t, 1.5, 1.5)
        t += 0.1
    # 왼쪽만 급감 (glitch) — EDGE 아님
    for _ in range(10):
        assert d.update(t, 0.4, 1.5) != d.EDGE
        t += 0.1


def test_edge_staggered_outside_window():
    d = _mk_edge()
    t = 0.0
    for _ in range(5):
        d.update(t, 1.5, 1.5)
        t += 0.1
    # 왼쪽 급감
    d.update(t, 0.4, 1.5)
    # 오른쪽은 0.5s 뒤 급감 — sync 창(0.3s) 밖
    t += 0.5
    r1 = d.update(t, 0.4, 0.4)
    assert r1 != d.EDGE, "시간차가 sync 창 밖이면 즉시 EDGE 금지"


def test_edge_staggered_inside_window():
    d = _mk_edge()
    t = 0.0
    for _ in range(5):
        d.update(t, 1.5, 1.5)
        t += 0.1
    d.update(t, 0.4, 1.5)          # L 급감 @ t
    t += 0.2                        # 0.2s 뒤 (창 내)
    assert d.update(t, 0.4, 0.4) == d.NONE   # R 급감, confirm 1/2
    t += 0.1
    assert d.update(t, 0.4, 0.4) == d.EDGE


def test_edge_glitch_recovers():
    """한쪽이 급감했다 복귀하면 플래그 해제 — 이후 정상 동작."""
    d = _mk_edge()
    t = 0.0
    for _ in range(5):
        d.update(t, 1.5, 1.5)
        t += 0.1
    d.update(t, 0.4, 1.5)           # L glitch
    t += 0.1
    d.update(t, 1.5, 1.5)           # L 복귀 → 플래그 해제
    t += 0.1
    # 이후 진짜 동시 급감
    d.update(t, 0.4, 0.4)
    t += 0.1
    assert d.update(t, 0.4, 0.4) == d.EDGE


def test_edge_stale_faults():
    d = _mk_edge()
    assert d.update(0.0, None, 1.5) == d.FAULT
    assert d.update(0.1, 1.5, None) == d.FAULT


def test_edge_slow_change_tracks_baseline():
    """서서히 변하는 바닥 거리(경사)는 baseline 추종 — 오탐 없음."""
    d = _mk_edge()
    t, dist = 0.0, 1.5
    for _ in range(30):
        assert d.update(t, dist, dist) == d.NONE
        dist -= 0.02   # 2cm/tick 완만한 감소 (thresh 0.30 미만)
        t += 0.1


# =============================================================================
# ReleaseDetector (다이어그램 TD6)
# =============================================================================
def test_release_confirm():
    r = ReleaseDetector(threshold_m=0.05, confirm_n=2)
    assert not r.update(0.20, 0.20)
    assert not r.update(0.04, 0.04)   # 1/2
    assert r.update(0.03, 0.04)       # 2/2 → 안착


def test_release_one_side_blocks():
    r = ReleaseDetector(threshold_m=0.05, confirm_n=2)
    for _ in range(5):
        assert not r.update(0.03, 0.20)   # 오른쪽 미달


def test_release_stale_resets():
    r = ReleaseDetector(threshold_m=0.05, confirm_n=2)
    r.update(0.03, 0.03)
    assert not r.update(None, 0.03)
    assert not r.update(0.03, 0.03)   # 다시 1/2


# =============================================================================
# 레이저 파서 / 리더 주입
# =============================================================================
def test_parse_tfmini_channels():
    out = parse_laser_line("L1 70cm strength=2178")
    assert out == [(1, pytest.approx(0.70), 2178)]
    out = parse_laser_line("L2 120cm strength=900")
    assert out == [(2, pytest.approx(1.20), 900)]


def test_parse_dual_channel_one_line():
    out = parse_laser_line("L1 70cm strength=2178 L2 65cm strength=1900")
    assert len(out) == 2
    assert out[0][0] == 1 and out[1][0] == 2


def test_parse_named_and_bare():
    assert parse_laser_line("dist: 350mm") == [(None, pytest.approx(0.35), None)]
    assert parse_laser_line("142") == [(None, pytest.approx(1.42), None)]


def test_dual_reader_inject_snapshot():
    r = DualLaserReader(wiring="single_port", median_window=3)
    r.inject_sample(1, 1.50)
    r.inject_sample(2, 1.48)
    sl, sr, _, _ = r.snapshot(stale_after_s=1.0)
    assert sl is not None and sl.distance_m == pytest.approx(1.50)
    assert sr is not None and sr.distance_m == pytest.approx(1.48)


def test_dual_reader_invalid_rejected():
    """0/1/2cm 급 무효값 (min_m=0.02 기본: 0.01 은 거부)."""
    r = DualLaserReader(wiring="single_port")
    r.inject_sample(1, 0.01)
    sl, _, _, _ = r.snapshot(stale_after_s=1.0)
    assert sl is None
