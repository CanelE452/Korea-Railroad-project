"""
Isaac Sim Replicator - DOPE 학습용 플라스틱 팔레트 합성 데이터 생성 (v3)

v3 변경점 (연구 가이드 v3 섹션 3.6, 3.7 기준):
  - 배경: warehouse.usd 로드 시도 -> 실패 시 바닥/벽 텍스처 다양화 fallback
  - Distractor: Isaac Sim Props (골판지 박스, 배럴, 콘) 시도 -> 실패 시 박스 위주 primitive
  - 카메라: 리프터 시점 제약 (높이 0.8-1.5m, pitch -30~+10, distance 1-4m)
  - 조명: RectLight 2-3개 + 랜덤 ON/OFF + 그림자
  - 팔레트 배치: z=0 고정, tilt <=5, yaw만 자유
  - 렌더 설정: RTL 기본 (대량 생성용)

사용법:
    conda activate pallet-pose
    python scripts/data_prep/isaac_sim/gen_replicator_data.py --num_frames 15000 --output_dir data/pallet/training_data

    # RayTracedLighting (대량 생성용, 기본)
    python scripts/data_prep/isaac_sim/gen_replicator_data.py --num_frames 50000

    # PathTracing (보강용, 고품질)
    python scripts/data_prep/isaac_sim/gen_replicator_data.py --renderer PathTracing --num_frames 5000

필수 환경변수:
    export OMNI_KIT_ACCEPT_EULA=YES
"""

import argparse
import os
import struct
import sys

# --- argparse를 SimulationApp 생성 전에 파싱 (renderer 선택 필요) ---
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--renderer", type=str, default="RayTracedLighting",
                         choices=["PathTracing", "RayTracedLighting"])
_pre_parser.add_argument("--seed", type=int, default=42)
_pre_parser.add_argument("--hdri_dir", type=str, default=None,
                         help="HDRI 파일 디렉토리 (dome light 배경용)")
_pre_args, _ = _pre_parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
    "anti_aliasing": 3 if _pre_args.renderer == "RayTracedLighting" else 0,
    # RTL: 3=FXAA (에지 계단현상 제거, 해상도 왜곡 없음)
    # PT:  0=Off (누적 샘플링으로 자체 AA 효과)
    "renderer": _pre_args.renderer,
})

import carb
settings = carb.settings.get_settings()

# === 렌더러 모드별 설정 ===
_chosen_renderer = _pre_args.renderer

if _chosen_renderer == "PathTracing":
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/ecoMode/enabled", False)
    settings.set("/rtx/pathtracing/spp", 32)
    settings.set("/rtx/pathtracing/totalSpp", 32)
    settings.set("/rtx/pathtracing/clampSpp", 32)
    settings.set("/rtx/pathtracing/maxBounces", 4)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/post/denoiser/enabled", False)
    print(f"[RENDER] PathTracing mode: spp=64, subframes=32, OptiX denoiser=ON")
elif _chosen_renderer == "RayTracedLighting":
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/ecoMode/enabled", False)
    settings.set("/rtx/directLighting/sampledLighting/enabled", True)
    settings.set("/rtx/ambientOcclusion/enabled", True)
    settings.set("/rtx/ambientOcclusion/rayLength", 1.0)
    settings.set("/rtx/reflections/enabled", True)
    settings.set("/rtx/reflections/maxRoughness", 1.0)
    settings.set("/rtx/indirectDiffuse/enabled", True)
    settings.set("/rtx/translucency/enabled", True)
    settings.set("/rtx/shadows/enabled", True)
    settings.set("/rtx/shadows/sampleCount", 8)      # 4->8: 그림자 경계 부드럽게
    settings.set("/rtx/post/denoiser/enabled", True)  # RTL GI/AO 노이즈 제거
    print(f"[RENDER] RayTracedLighting mode: AO+reflections+shadows ON, shadow samples=8, denoiser=ON, FXAA=ON")

_current_mode = settings.get("/rtx/rendermode")
print(f"[RENDER] rendermode = {_current_mode}")

# === 공통 설정 ===
settings.set("/rtx/post/dlss/enabled", False)
settings.set("/rtx/post/dlss/execMode", 0)
if _chosen_renderer == "RayTracedLighting":
    # RTL: FXAA 활성화 (SimulationApp anti_aliasing=3과 연동)
    settings.set("/rtx/post/aa/op", 3)       # 3=FXAA
    settings.set("/rtx/post/aa/enabled", True)
else:
    # PT: AA 불필요 (누적 샘플링으로 자체 AA)
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/post/aa/enabled", False)
settings.set("/rtx/resourcemanager/enableTextureStreaming", False)
settings.set("/rtx/renderScaleFactor", 1.0)
settings.set("/app/hydra/aperture/conform", 0)
settings.set("/rtx/upscaler/enabled", False)
_aa_mode = "FXAA" if _chosen_renderer == "RayTracedLighting" else "Off"
print(f"[RENDER] AA={_aa_mode}, DLSS=Off, upscaler=Off, renderScale=1.0")

# --- 톤매핑 ---
settings.set("/rtx/post/tonemap/op", 4)            # 4=Reinhard
settings.set("/rtx/post/histogram/enabled", False)
settings.set("/rtx/post/tonemap/filmIso", 300.0)
settings.set("/rtx/post/tonemap/cameraShutter", 60.0)
settings.set("/rtx/post/tonemap/fNumber", 3.5)
settings.set("/rtx/post/tonemap/whitepoint", 8.0)

# --- 후처리 비활성화 ---
settings.set("/rtx/post/chromaticAberration/enabled", False)
settings.set("/rtx/post/lensFlare/enabled", False)
settings.set("/rtx/post/motionblur/enabled", False)
settings.set("/rtx/post/bloom/enabled", False)
settings.set("/rtx/post/sharpen/enabled", False)    # v3: sharpen OFF (RTL 기본, 에일리어싱 강조 방지)

# Firefly 필터
settings.set("/rtx/pathtracing/fireflyClampingEnabled", True)
settings.set("/rtx/pathtracing/fireflyClampingThreshold", 50.0)
# 텍스처 스트리밍 안정화 + 동기 렌더링 강제
settings.set("/rtx/materialDb/syncLoads", True)
settings.set("/rtx/hydra/materialSyncLoads", True)
settings.set("/omni.kit.plugin/syncUsdLoads", True)
settings.set("/app/asyncRendering", False)
settings.set("/rtx/hydra/cacheSyncLoads", True)

import numpy as np
import omni.replicator.core as rep
import omni.usd
from sdg_config import *
from sdg_math import (euler_to_rotation_matrix, rotation_matrix_to_quat_xyzw,
                       rotation_matrix_to_euler_deg, build_camera_matrix,
                       build_view_matrix, _canonical_corners)
from sdg_annotation import _compute_visibility, write_ndds_json
from sdg_usd_xform import (
    _distractor_prim_path_cache, _rep_prim_path_cache, _xformable_cache,
    _resolve_rep_prim_path, _resolve_distractor_prim_path,
    _set_pose_usd, _set_pose_usd_rep, _set_camera_look_at_usd,
    _randomize_lights_usd, _set_light_attrs_usd,
    _set_distractor_visible, _apply_distractor_color,
)
from sdg_distractors import (
    _select_camera_mode, _sample_camera_pose, _jitter_look_at,
    _sample_floor_distractor_pos, _randomize_distractors,
)
from sdg_scene import (
    compute_model_info, convert_all_gltf,
    _pick_pallet_color, setup_scene, register_randomizers,
    _apply_color_to_all_materials, _change_floor_wall_textures,
    _pick_weighted_texture,
)



# ============================================================
# 데이터 생성
# ============================================================
def generate_data(render_product, pallet_prims, camera, output_dir: str,
                  num_frames: int, pallet_prim_paths: list, K: np.ndarray,
                  prebuilt: dict, seed: int = 42,
                  image_width: int = IMAGE_WIDTH,
                  image_height: int = IMAGE_HEIGHT):
    os.makedirs(output_dir, exist_ok=True)

    rt_subframes = RT_SUBFRAMES_PT if _chosen_renderer == "PathTracing" else RT_SUBFRAMES_RTL

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])
    print(f"[INFO] Annotator direct capture -> {output_dir}")
    print(f"[INFO] renderer={_chosen_renderer}, rt_subframes={rt_subframes}")

    # DLSS 최종 확인
    _dlss_state = settings.get("/rtx/post/dlss/enabled")
    _aa_state = settings.get("/rtx/post/aa/op")
    _scale_state = settings.get("/rtx/renderScaleFactor")
    print(f"[RENDER] generate_data entry: DLSS={_dlss_state}, AA_op={_aa_state}, renderScale={_scale_state}")
    if _dlss_state:
        print("[RENDER] WARNING: DLSS still active! Force disabling...")
        settings.set("/rtx/post/dlss/enabled", False)
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/aa/enabled", False)
        settings.set("/rtx/renderScaleFactor", 1.0)

    # warm-up
    for _ in range(WARMUP_STEPS):
        rep.orchestrator.step(rt_subframes=rt_subframes)
        rep.orchestrator.wait_until_complete()
    _ = rgb_annot.get_data()
    print(f"  warm-up done ({WARMUP_STEPS} steps)")

    rng = np.random.default_rng(seed)
    stage = omni.usd.get_context().get_stage()
    distractor_prims = prebuilt["distractor_prims"]
    distractor_sizes = prebuilt["distractor_sizes"]
    use_props = prebuilt["use_props"]

    MIN_VISIBILITY = 0.6
    MAX_RETRIES = 5

    # v9: plane prim 캐싱 — stage.Traverse()를 매 프레임 호출하지 않도록
    from pxr import UsdGeom as _UG_plane
    _cached_plane_prims = []
    for _fp in stage.Traverse():
        _fp_path = str(_fp.GetPath())
        if "/Replicator/Plane" in _fp_path and _fp.GetTypeName() == "Xform":
            _cached_plane_prims.append(_fp)
    print(f"  [CACHE] {len(_cached_plane_prims)} plane prims cached")

    print(f"\n[START] generating {num_frames} frames...")
    saved = 0
    attempts = 0
    # 진단 모드: --diagnose 플래그 시 모델 순서대로 1장씩
    _diagnose_mode = os.environ.get("DIAGNOSE_MODELS", "0") == "1"
    while saved < num_frames:
        attempts += 1
        if _diagnose_mode:
            idx = saved % len(pallet_prims)
        else:
            idx = rng.integers(len(pallet_prims))
        pallet, s, base_rot, bbox_min, bbox_max, z_offset, R_canonical, cbbox_min, cbbox_max = pallet_prims[idx]

        # 팔레트 색상
        color = _pick_pallet_color(rng)
        _apply_color_to_all_materials(stage, color)

        # v3: 팔레트 pose - z=0 고정, tilt <=5, yaw만 자유
        tilt_x = float(rng.uniform(-PALLET_TILT_MAX, PALLET_TILT_MAX))
        tilt_y = float(rng.uniform(-PALLET_TILT_MAX, PALLET_TILT_MAX))
        pallet_pos = (
            float(rng.uniform(-0.3, 0.3)),
            float(rng.uniform(-0.3, 0.3)),
            float(z_offset),                  # v3: z=0 고정 (z_offset만, 추가 랜덤 없음)
        )
        pallet_rot = (
            base_rot[0] + tilt_x,             # v3: <=5 tilt
            base_rot[1] + tilt_y,
            base_rot[2] + float(rng.uniform(0, 360)),  # yaw만 자유
        )

        for j, (p, p_s, p_br, _, _, _, _, _, _) in enumerate(pallet_prims):
            if j == idx:
                with p:
                    rep.modify.pose(position=pallet_pos,
                                    rotation=pallet_rot, scale=(s, s, s))
            else:
                with p:
                    rep.modify.pose(position=(0, 0, -100),
                                    scale=(p_s, p_s, p_s))

        # v11: 카메라 - 3모드 (A: 리프터 마운트 60%, B: 높은 시점 25%, C: 바닥 15%)
        cam_pos = _sample_camera_pose(rng, pallet_pos, front_view=False)

        # look-at jitter (팔레트 위치 기준)
        # 전면 뷰: look-at 높이를 팔레트 중간(0.08m)으로 유지하여 수평 시선
        look_target = _jitter_look_at(rng, (
            pallet_pos[0] + LOOK_AT_TARGET[0],
            pallet_pos[1] + LOOK_AT_TARGET[1],
            LOOK_AT_TARGET[2],
        ))

        # 카메라 pose — rep.modify.pose() 유지 (look_at convention 호환성)
        with camera:
            rep.modify.pose(
                position=cam_pos,
                look_at=look_target,
            )

        # v7: 배경 3모드 — 창고 40% / 프로시저럴 실내 30% / 야외(HDRI) 30%
        _bg_roll = float(rng.random())
        _use_warehouse_this_frame = (
            prebuilt["warehouse_loaded"]
            and _bg_roll < WAREHOUSE_BG_PROBABILITY
        )
        _use_outdoor_this_frame = (
            not _use_warehouse_this_frame
            and prebuilt.get("hdri_files")
            and float(rng.random()) < OUTDOOR_BG_PROBABILITY / (1.0 - WAREHOUSE_BG_PROBABILITY)
        )

        _warehouse_prim = stage.GetPrimAtPath("/World/Warehouse")
        if _warehouse_prim and _warehouse_prim.IsValid():
            from pxr import UsdGeom as _UG_toggle
            _UG_toggle.Imageable(_warehouse_prim).MakeVisible() if _use_warehouse_this_frame \
                else _UG_toggle.Imageable(_warehouse_prim).MakeInvisible()

        # 프로시저럴 바닥/벽 표시 제어 (v9: 캐시된 plane prims 사용)
        for _fp in _cached_plane_prims:
            if _use_warehouse_this_frame or _use_outdoor_this_frame:
                _UG_plane.Imageable(_fp).MakeInvisible()
            else:
                _UG_plane.Imageable(_fp).MakeVisible()

        # 프로시저럴 배경일 때 텍스처 랜덤 변경 (현실적 70% / 비현실적 30%)
        if not _use_warehouse_this_frame:
            _tex_real = prebuilt["tex_realistic"]
            _tex_style = prebuilt["tex_stylized"]
            _floor_tex = _pick_weighted_texture(rng, _tex_real, _tex_style, 0.7)
            if not _use_outdoor_this_frame:
                _wall_tex = _pick_weighted_texture(rng, _tex_real, _tex_style, 0.7)
            else:
                _wall_tex = _floor_tex  # 야외에서는 벽 없으니 무의미
            _change_floor_wall_textures(stage, _floor_tex, _wall_tex)

        # v6: 팔레트 상면 중심 계산 — annotation과 동일한 방식 사용
        # R_random = R_pallet @ R_canonical^T, 이후 canonical corners를 world로 변환
        _R_pallet_mat = euler_to_rotation_matrix(pallet_rot)
        _R_random = _R_pallet_mat @ R_canonical.T
        _cbmin = np.array(cbbox_min)
        _cbmax = np.array(cbbox_max)
        _corners_c = _canonical_corners(_cbmin, _cbmax)
        _corners_world = (s * (_R_random @ _corners_c.T)).T + np.array(pallet_pos)
        # 상면(top face) = corners 0,1,4,5 (Y_max in canonical = 위쪽)
        _top_face_center = _corners_world[[0, 1, 4, 5]].mean(axis=0)
        _pallet_surface_center = (
            float(_top_face_center[0]),
            float(_top_face_center[1]),
            float(_top_face_center[2]),  # 실제 상면 Z 좌표 사용
        )

        # v6: 디스트랙터 랜덤 배치 (팔레트 위 적재 포함)
        _pallet_half_extents = (
            (_cbmax[0] - _cbmin[0]) * s / 2,  # medium/2 (canonical X)
            (_cbmax[2] - _cbmin[2]) * s / 2,  # long/2 (canonical Z)
        )
        _randomize_distractors(rng, distractor_prims, stage, use_props=use_props,
                               pallet_pos=_pallet_surface_center,
                               pallet_yaw_deg=pallet_rot[2],
                               pallet_half_extents=_pallet_half_extents,
                               distractor_sizes=distractor_sizes,
                               cam_pos=cam_pos,
                               distractor_shader_cache=prebuilt.get("distractor_shader_cache", []))

        # 렌더링
        rep.orchestrator.step(rt_subframes=rt_subframes)
        rep.orchestrator.wait_until_complete()

        # visibility 사전 검사 (canonical bbox 사용)
        visibility = _compute_visibility(
            cam_pos, look_target, K, image_width, image_height,
            cbbox_min, cbbox_max, pallet_pos, pallet_rot, s,
            R_canonical=R_canonical,
        )
        if visibility < MIN_VISIBILITY:
            continue

        # RGB 캡처 및 저장
        rgb_data = rgb_annot.get_data()
        if rgb_data is None or rgb_data.size == 0:
            print(f"  [WARN] frame {saved}: no RGB data, skipping")
            continue

        # 밝기 검증
        mean_brightness = float(rgb_data[:, :, :3].mean())
        if mean_brightness < 40:
            print(f"  [SKIP] frame too dark (mean={mean_brightness:.1f})")
            continue
        if mean_brightness > 240:
            print(f"  [SKIP] frame too bright (mean={mean_brightness:.1f})")
            continue

        rgb_path = os.path.join(output_dir, f"{saved:06d}.png")
        _save_rgba_as_png(rgb_data, rgb_path)

        json_path = os.path.join(output_dir, f"{saved:06d}.json")
        write_ndds_json(
            json_path, cam_pos, look_target,
            K, image_width, image_height,
            cbbox_min, cbbox_max, pallet_pos, pallet_rot, s,
            R_canonical=R_canonical,
        )

        saved += 1
        _usd_paths = prebuilt.get("usd_paths", [])
        _model_name = os.path.basename(_usd_paths[idx]) if idx < len(_usd_paths) else f"idx={idx}"
        if _diagnose_mode or saved % 10 == 0 or saved == num_frames:
            print(f"  [{saved}/{num_frames}] model={_model_name} base_rot={base_rot} attempts={attempts}")
        # v10: Python GC — 50프레임마다 가비지 컬렉션
        if saved % 50 == 0:
            import gc
            gc.collect()

    rgb_annot.detach()
    rep.orchestrator.wait_until_complete()
    print(f"\n[DONE] {num_frames} frames -> {output_dir} (total attempts: {attempts})")


# ============================================================
# PNG 저장
# ============================================================
try:
    import cv2 as _cv2
    def _save_rgba_as_png(rgba_array, filepath):
        bgr = _cv2.cvtColor(rgba_array[:, :, :3], _cv2.COLOR_RGB2BGR)
        _cv2.imwrite(filepath, bgr)
except ImportError:
    def _save_rgba_as_png(rgba_array, filepath):
        import zlib
        h, w = rgba_array.shape[:2]
        rgb = rgba_array[:, :, :3]
        raw = b""
        for y in range(h):
            raw += b"\x00" + rgb[y].tobytes()
        compressed = zlib.compress(raw)

        def _chunk(chunk_type, data):
            c = chunk_type + data
            crc = zlib.crc32(c) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

        with open(filepath, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
            f.write(_chunk(b"IDAT", compressed))
            f.write(_chunk(b"IEND", b""))


# ============================================================
# 테스트 시나리오 생성
# ============================================================
def generate_test_scenarios(render_product, pallet_prims, camera, output_dir: str,
                            pallet_prim_paths: list, K: np.ndarray,
                            prebuilt: dict, seed: int = 42,
                            image_width: int = IMAGE_WIDTH,
                            image_height: int = IMAGE_HEIGHT):
    """v3: 리프터 시점 테스트 시나리오. 다양한 거리/높이/각도 조합."""
    os.makedirs(output_dir, exist_ok=True)

    rt_subframes = RT_SUBFRAMES_PT if _chosen_renderer == "PathTracing" else RT_SUBFRAMES_RTL

    stage = omni.usd.get_context().get_stage()
    num_pallets = len(pallet_prims)

    # v3: 리프터 시점 제약 내의 테스트 시나리오
    scenarios = []

    # 근거리 (1-2m), 다양한 높이/yaw
    for dist, height, yaw_deg, pallet_yaw in [
        (1.2, 1.0, 0,   0),
        (1.5, 0.8, -20, 45),
        (1.8, 1.2, 15,  90),
        (1.5, 1.4, -30, 180),
        (1.3, 0.9, 30,  270),
    ]:
        scenarios.append((f"close_{dist}m_h{height}_y{yaw_deg}", "lifter",
                          (dist, height, yaw_deg, pallet_yaw)))

    # 중거리 (2-3m)
    for dist, height, yaw_deg, pallet_yaw in [
        (2.5, 1.0, 0,   0),
        (3.0, 1.2, -15, 60),
        (2.8, 0.9, 20,  120),
        (2.2, 1.3, -25, 240),
        (2.7, 1.1, 10,  300),
    ]:
        scenarios.append((f"mid_{dist}m_h{height}_y{yaw_deg}", "lifter",
                          (dist, height, yaw_deg, pallet_yaw)))

    # 원거리 (3-4m)
    for dist, height, yaw_deg, pallet_yaw in [
        (3.5, 1.0, 0,   0),
        (4.0, 1.2, -10, 90),
        (3.8, 0.8, 5,   180),
        (3.2, 1.5, -20, 45),
        (3.7, 1.1, -5,  270),
    ]:
        scenarios.append((f"far_{dist}m_h{height}_y{yaw_deg}", "lifter",
                          (dist, height, yaw_deg, pallet_yaw)))

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])

    for _ in range(WARMUP_STEPS):
        rep.orchestrator.step(rt_subframes=rt_subframes)
        rep.orchestrator.wait_until_complete()
    _ = rgb_annot.get_data()

    rng = np.random.default_rng(seed)

    print(f"\n[TEST] generating {len(scenarios)} test scenarios...", flush=True)
    for i, (label, stype, params) in enumerate(scenarios):
        idx = i % num_pallets
        pallet, s, base_rot, bbox_min, bbox_max, z_offset, R_canonical, cbbox_min, cbbox_max = pallet_prims[idx]

        # v12: USD API 직접 pose 설정
        for j, (p, p_s, _, _, _, _, _, _, _) in enumerate(pallet_prims):
            if j != idx:
                _set_pose_usd_rep(stage, p, position=(0, 0, -100),
                                  scale=(p_s, p_s, p_s))

        color = _pick_pallet_color(rng)
        _apply_color_to_all_materials(stage, color)

        dist, height, yaw_deg, pallet_yaw = params
        yaw_rad = np.radians(yaw_deg)
        cam_pos = (dist * np.cos(yaw_rad), dist * np.sin(yaw_rad), height)
        look_target = (0.0, 0.0, 0.08)

        pallet_pos = (0.0, 0.0, z_offset)
        pallet_rot = (base_rot[0], base_rot[1], base_rot[2] + pallet_yaw)
        _set_pose_usd_rep(stage, pallet, position=pallet_pos,
                          rotation_deg=pallet_rot, scale=(s, s, s))

        with camera:
            rep.modify.pose(position=cam_pos, look_at=look_target)

        rep.orchestrator.step(rt_subframes=rt_subframes)
        rep.orchestrator.wait_until_complete()

        rgb_data = rgb_annot.get_data()
        if rgb_data is not None and rgb_data.size > 0:
            rgb_path = os.path.join(output_dir, f"{i:06d}.png")
            _save_rgba_as_png(rgb_data, rgb_path)
            print(f"  [{i}] {label}: cam=({cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f}) -> {rgb_path}", flush=True)
        else:
            print(f"  [{i}] {label}: WARNING - no RGB data!", flush=True)

        json_path = os.path.join(output_dir, f"{i:06d}.json")
        write_ndds_json(
            json_path, cam_pos, look_target,
            K, image_width, image_height,
            cbbox_min, cbbox_max, pallet_pos, pallet_rot, s,
            R_canonical=R_canonical,
        )

    rgb_annot.detach()
    print(f"\n[DONE] {len(scenarios)} test frames -> {output_dir}", flush=True)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Replicator - DOPE pallet training data (v3)")
    parser.add_argument("--num_frames", type=int, default=15000)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gltf_dir", type=str, default=GLTF_DIR)
    parser.add_argument("--usd_cache_dir", type=str, default=USD_CACHE_DIR)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--renderer", type=str, default="RayTracedLighting",
                        choices=["PathTracing", "RayTracedLighting"],
                        help="renderer (default: RTL for bulk, PathTracing for quality)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdri_dir", type=str, default=None,
                        help="HDRI directory for dome light background")
    parser.add_argument("--test_mode", action="store_true",
                        help="lifter-view test scenarios")
    parser.add_argument("--overlay", action="store_true",
                        help="auto-run visualize_annotations.py after generation")
    args = parser.parse_args()

    image_width = args.width
    image_height = args.height

    print("=" * 60)
    print("Isaac Sim Replicator - Pallet SDG v3")
    print(f"  renderer: {_chosen_renderer}")
    print(f"  seed: {args.seed}")
    print(f"  rt_subframes: {RT_SUBFRAMES_PT if _chosen_renderer == 'PathTracing' else RT_SUBFRAMES_RTL}")
    print(f"  camera v11: Mode A {int(CAMERA_MODE_PROBS[0]*100)}% lifter-mount "
          f"(h={CAMERA_CONSTRAINTS['height_min']}-{CAMERA_CONSTRAINTS['height_max']}m, "
          f"dist={CAMERA_CONSTRAINTS['distance_min']}-{CAMERA_CONSTRAINTS['distance_max']}m)")
    print(f"  camera v11: Mode B {int(CAMERA_MODE_PROBS[1]*100)}% high-view "
          f"(h={CAMERA_MODE_B['height_min']}-{CAMERA_MODE_B['height_max']}m)")
    print(f"  camera v11: Mode C {int(CAMERA_MODE_PROBS[2]*100)}% ground-level "
          f"(h={FRONT_VIEW_CONSTRAINTS['height_min']}-{FRONT_VIEW_CONSTRAINTS['height_max']}m)")
    print(f"  background: warehouse {int(WAREHOUSE_BG_PROBABILITY*100)}%, outdoor {int(OUTDOOR_BG_PROBABILITY*100)}%"
          f", indoor {int((1-WAREHOUSE_BG_PROBABILITY)*(1-OUTDOOR_BG_PROBABILITY)*100)}%")
    print(f"  cargo occlusion: {int(CARGO_ON_PALLET_PROBABILITY*100)}%")
    print(f"  pallet tilt: <={PALLET_TILT_MAX}deg")
    print("=" * 60)

    # 1. glTF -> USD
    print("\n[Step 1] glTF -> USD conversion")
    usd_paths = convert_all_gltf(args.gltf_dir, GLTF_FILES, args.usd_cache_dir)
    if not usd_paths:
        print("[ERROR] No USD files. Check glTF paths.")
        simulation_app.close()
        sys.exit(1)

    # 2. 각 모델의 스케일 + 눕히기 회전 + z_offset 계산
    print("\n[Step 2] Computing per-model scale, orientation & z_offset")
    model_infos = []
    for usd_path in usd_paths:
        info = compute_model_info(usd_path)
        model_infos.append(info)

    # 3. Scene setup
    print(f"\n[Step 3] Scene setup ({image_width}x{image_height})")
    pallet_prims, camera, render_product, pallet_prim_paths, K, prebuilt = setup_scene(
        usd_paths, model_infos, image_width, image_height, hdri_dir=args.hdri_dir,
        chosen_renderer=_chosen_renderer, settings=settings,
    )
    print(f"  {len(pallet_prims)} pallets loaded")
    print(f"  {len(prebuilt['distractor_prims'])} distractor pool, {MAX_DISTRACTORS_PER_FRAME}/frame ({'USD Props' if prebuilt['use_props'] else 'primitive fallback'})")
    print(f"  background: {'warehouse.usd' if prebuilt['warehouse_loaded'] else 'floor/wall fallback'}")

    # 4. Domain Randomization
    print("\n[Step 4] Domain Randomization (v3)")
    register_randomizers(pallet_prims, camera, prebuilt)
    print("  v12: OmniGraph randomizers removed - all DR via USD API in generate_data loop")

    # 5. Generate data
    if args.test_mode:
        print("\n[Step 5] Test scenario generation (lifter-view)")
        generate_test_scenarios(render_product, pallet_prims, camera, args.output_dir,
                                pallet_prim_paths, K, prebuilt, seed=args.seed,
                                image_width=image_width, image_height=image_height)
    else:
        print("\n[Step 5] Data generation")
        generate_data(render_product, pallet_prims, camera, args.output_dir,
                      args.num_frames, pallet_prim_paths, K, prebuilt, seed=args.seed,
                      image_width=image_width, image_height=image_height)

    simulation_app.close()
    print("\n[DONE] Isaac Sim closed")

    # 6. Auto overlay visualization
    if args.overlay:
        overlay_dir = os.path.join(args.output_dir, "overlay")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vis_script = os.path.join(script_dir, "..", "visualize_annotations.py")
        if os.path.isfile(vis_script):
            print(f"\n[Step 6] Running annotation overlay -> {overlay_dir}")
            import subprocess
            ret = subprocess.run(
                [sys.executable, vis_script,
                 "--data_dir", args.output_dir,
                 "--output_dir", overlay_dir],
                check=False,
            )
            if ret.returncode == 0:
                print(f"  overlay done: {overlay_dir}")
            else:
                print(f"  [WARN] overlay failed (exit code {ret.returncode})")
        else:
            print(f"  [WARN] visualize_annotations.py not found at {vis_script}")


if __name__ == "__main__":
    main()
