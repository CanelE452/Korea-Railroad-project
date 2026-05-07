"""디스트랙터 배치 및 카메라 포즈 샘플링 함수.

gen_replicator_data.py에서 분리된 디스트랙터/카메라 관련 로직.
"""

import numpy as np
from sdg_config import (
    CAMERA_MODE_PROBS, CAMERA_CONSTRAINTS, CAMERA_MODE_B,
    FRONT_VIEW_CONSTRAINTS, LOOK_AT_JITTER,
    MAX_DISTRACTORS_PER_FRAME, CARGO_ON_PALLET_PROBABILITY,
    CARGO_OCCLUSION_TIERS, CARGO_Z_CLEARANCE,
)
from sdg_usd_xform import _set_pose_usd_rep, _apply_distractor_color


# ============================================================
# v11: 리프터 시점 카메라 샘플링 (3모드)
# ============================================================
def _select_camera_mode(rng):
    """v11: 3모드 중 하나를 확률적으로 선택.
    Returns: 'A' (리프터 마운트), 'B' (높은 시점), 'C' (바닥 레벨)
    """
    r = rng.random()
    if r < CAMERA_MODE_PROBS[0]:
        return 'A'
    elif r < CAMERA_MODE_PROBS[0] + CAMERA_MODE_PROBS[1]:
        return 'B'
    return 'C'

def _sample_camera_pose(rng, pallet_pos=(0, 0, 0), front_view=False):
    """v11: 3모드 카메라 위치 생성.

    Args:
        front_view: True이면 강제 Mode C (하위 호환성 유지)

    반환: (cam_x, cam_y, cam_z)
    """
    if front_view:
        cc = FRONT_VIEW_CONSTRAINTS
    else:
        mode = _select_camera_mode(rng)
        if mode == 'A':
            cc = CAMERA_CONSTRAINTS       # 리프터 마운트
        elif mode == 'B':
            cc = CAMERA_MODE_B            # 높은 시점
        else:
            cc = FRONT_VIEW_CONSTRAINTS   # 바닥 레벨
    distance = float(rng.uniform(cc["distance_min"], cc["distance_max"]))
    height = float(rng.uniform(cc["height_min"], cc["height_max"]))
    yaw_deg = float(rng.uniform(cc["yaw_min"], cc["yaw_max"]))

    # 팔레트 yaw에 상대적인 카메라 방위각 (팔레트 정면 기준 +-45도)
    # 여기에 팔레트 전방 방향은 별도로 계산하지 않고, 절대 yaw로 처리
    # (팔레트 yaw가 360도 랜덤이므로 상대/절대 구분 무의미)
    yaw_rad = np.radians(yaw_deg)

    cam_x = float(pallet_pos[0]) + distance * np.cos(yaw_rad)
    cam_y = float(pallet_pos[1]) + distance * np.sin(yaw_rad)
    cam_z = height

    return (cam_x, cam_y, cam_z)


def _jitter_look_at(rng, base_target):
    return (
        base_target[0] + float(rng.uniform(-LOOK_AT_JITTER, LOOK_AT_JITTER)),
        base_target[1] + float(rng.uniform(-LOOK_AT_JITTER, LOOK_AT_JITTER)),
        base_target[2] + float(rng.uniform(-LOOK_AT_JITTER * 0.5, LOOK_AT_JITTER * 0.5)),
    )


def _sample_floor_distractor_pos(rng, pallet_pos, cam_pos, max_attempts=10):
    """바닥 디스트랙터 위치를 샘플링하되, 카메라-팔레트 시선 경로를 피한다.
    XY 평면에서 카메라→팔레트 방향 벡터에 대한 수직 거리가 충분해야 통과."""
    CORRIDOR_HALF_WIDTH = 0.8  # 시선 경로 좌우 0.8m 이내는 거부
    for _ in range(max_attempts):
        x = float(rng.uniform(-3.0, 3.0))
        y = float(rng.uniform(-3.0, 3.0))
        if cam_pos is not None and pallet_pos is not None:
            # XY 평면에서 카메라→팔레트 벡터
            cx, cy = cam_pos[0], cam_pos[1]
            px, py = pallet_pos[0], pallet_pos[1]
            dx, dy = px - cx, py - cy
            seg_len = np.sqrt(dx * dx + dy * dy)
            if seg_len > 0.01:
                # 디스트랙터 점을 카메라→팔레트 선분에 투영
                tx = x - cx
                ty = y - cy
                t_proj = (tx * dx + ty * dy) / (seg_len * seg_len)
                # t_proj ∈ [0, 1]이면 카메라~팔레트 사이에 있음
                if 0.0 < t_proj < 1.0:
                    # 선분까지의 수직 거리
                    perp_x = tx - t_proj * dx
                    perp_y = ty - t_proj * dy
                    perp_dist = np.sqrt(perp_x * perp_x + perp_y * perp_y)
                    if perp_dist < CORRIDOR_HALF_WIDTH:
                        continue  # 시선 경로 내 → 거부, 재샘플링
        return (x, y, float(rng.uniform(0.0, 0.05)))
    # max_attempts 소진 → 팔레트 뒤쪽에 배치 (카메라 반대편)
    if cam_pos is not None and pallet_pos is not None:
        cx, cy = cam_pos[0], cam_pos[1]
        px, py = pallet_pos[0], pallet_pos[1]
        away_x = px + (px - cx) * 0.5 + float(rng.uniform(-1.0, 1.0))
        away_y = py + (py - cy) * 0.5 + float(rng.uniform(-1.0, 1.0))
        return (away_x, away_y, float(rng.uniform(0.0, 0.05)))
    return (float(rng.uniform(-3.0, 3.0)), float(rng.uniform(-3.0, 3.0)),
            float(rng.uniform(0.0, 0.05)))


# ============================================================
# v3: 디스트랙터 랜덤 배치
# ============================================================
def _randomize_distractors(rng, distractor_prims, stage, use_props=False,
                           pallet_pos=None, pallet_yaw_deg=0.0,
                           pallet_half_extents=(0.5, 0.6),
                           distractor_sizes=None, cam_pos=None,
                           distractor_shader_cache=None):
    """v5: 풀에서 랜덤 선택하여 3~10개 디스트랙터를 다양하게 배치.
    pallet_pos가 주어지면 일부를 팔레트 위에 적재.
    pallet_yaw_deg: 팔레트의 Z축 회전(world frame).
    pallet_half_extents: (medium/2, long/2) 팔레트 반경.
    cam_pos: 카메라 위치 — 바닥 디스트랙터가 시선 경로를 가리지 않도록 배치."""
    pool_size = len(distractor_prims)
    num_active = int(rng.integers(3, min(MAX_DISTRACTORS_PER_FRAME, pool_size) + 1))
    # 풀에서 랜덤으로 num_active개 인덱스 선택
    active_indices = list(rng.choice(pool_size, size=num_active, replace=False))

    # v7: 팔레트 위 적재 — 가림 분포 제어
    #   약한(50%) / 보통(33%) / 심한(17%), 50%+ 극단 가림 없음
    on_pallet_indices = set()
    cargo_scale_range = (0.15, 0.30)  # default: 약한
    if pallet_pos and float(rng.random()) < CARGO_ON_PALLET_PROBABILITY:
        # tier 선택 (누적 확률)
        r = float(rng.random())
        cumul = 0.0
        num_on_pallet = 1
        for prob, max_count, scale_range in CARGO_OCCLUSION_TIERS:
            cumul += prob
            if r < cumul:
                num_on_pallet = int(rng.integers(1, max_count + 1))
                cargo_scale_range = scale_range
                break
        num_on_pallet = min(num_on_pallet, len(active_indices))
        on_pallet_indices = set(active_indices[:num_on_pallet])
    active_indices = set(active_indices)

    for i, d in enumerate(distractor_prims):
        if i in active_indices:
            if i in on_pallet_indices and pallet_pos:
                # 팔레트 위에 배치 — pallet_pos는 이미 상면 중심 (v6)
                cargo_z = pallet_pos[2] + CARGO_Z_CLEARANCE
                # 팔레트 로컬 프레임에서 오프셋 → yaw 회전으로 world 변환
                hx, hz = pallet_half_extents  # medium/2, long/2
                local_dx = float(rng.uniform(-hx * 0.8, hx * 0.8))
                local_dy = float(rng.uniform(-hz * 0.8, hz * 0.8))
                yaw_rad = np.radians(pallet_yaw_deg)
                cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
                world_dx = cos_y * local_dx - sin_y * local_dy
                world_dy = sin_y * local_dx + cos_y * local_dy
                pos = (
                    pallet_pos[0] + world_dx,
                    pallet_pos[1] + world_dy,
                    cargo_z,
                )
                # v8: bbox 정규화 — target_size / original_max_dim
                target_min, target_max = cargo_scale_range
                target_size = float(rng.uniform(target_min, target_max))
                orig_dim = distractor_sizes[i] if distractor_sizes and i < len(distractor_sizes) else 0.5
                cargo_sc = np.clip(target_size / orig_dim, 0.05, 3.0)
                _cargo_rot = (
                    float(rng.uniform(-3, 3)),
                    float(rng.uniform(-3, 3)),
                    float(rng.uniform(0, 360)),
                )
                if use_props:
                    _set_pose_usd_rep(stage, d, position=pos, rotation_deg=_cargo_rot,
                                      scale=(cargo_sc, cargo_sc, cargo_sc))
                else:
                    # primitive: 약간의 비균일 스케일 허용
                    sc_var = float(rng.uniform(0.8, 1.2))
                    _set_pose_usd_rep(stage, d, position=pos, rotation_deg=_cargo_rot,
                                      scale=(cargo_sc, cargo_sc, cargo_sc * sc_var))
                print(f"    [CARGO] distractor {i} ON pallet target={target_size:.2f}m orig={orig_dim:.2f}m sc={cargo_sc:.2f} z={cargo_z:.2f}")
                continue  # skip normal placement below
            else:
                # 바닥에 배치 — 카메라-팔레트 시선 경로를 피해서 배치
                pos = _sample_floor_distractor_pos(rng, pallet_pos, cam_pos)
            # v3: 박스 형태에 맞는 비균일 스케일
            _floor_rot = (
                float(rng.uniform(-5, 5)),
                float(rng.uniform(-5, 5)),
                float(rng.uniform(0, 360)),
            )
            if use_props:
                sc = float(rng.uniform(0.8, 1.5))
                _set_pose_usd_rep(stage, d, position=pos, rotation_deg=_floor_rot,
                                  scale=(sc, sc, sc))
            else:
                sx = float(rng.uniform(0.2, 0.8))
                sy = float(rng.uniform(0.2, 0.8))
                sz = float(rng.uniform(0.15, 0.6))
                _set_pose_usd_rep(stage, d, position=pos, rotation_deg=_floor_rot,
                                  scale=(sx, sy, sz))
                # 디스트랙터 색상 변경 (primitive fallback)
                palette_type = rng.choice(["cardboard", "plastic", "concrete", "metal"])
                if palette_type == "cardboard":
                    color = (float(rng.uniform(0.45, 0.72)), float(rng.uniform(0.35, 0.55)), float(rng.uniform(0.18, 0.35)))
                elif palette_type == "plastic":
                    color = (float(rng.uniform(0.15, 0.90)), float(rng.uniform(0.15, 0.90)), float(rng.uniform(0.15, 0.85)))
                elif palette_type == "concrete":
                    base = float(rng.uniform(0.30, 0.55))
                    color = (base, base * float(rng.uniform(0.95, 1.05)), base * float(rng.uniform(0.90, 1.0)))
                else:  # metal
                    base = float(rng.uniform(0.40, 0.75))
                    color = (base, base, base * float(rng.uniform(0.95, 1.05)))
                _apply_distractor_color(stage, i, color, distractor_shader_cache, rng=rng)
        else:
            _set_pose_usd_rep(stage, d, position=(0, 0, -200))
