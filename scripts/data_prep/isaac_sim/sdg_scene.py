"""
씬 구성 함수 모듈 — gen_replicator_data.py에서 분리.

텍스처 생성, 모델 정보 계산, glTF→USD 변환, 씬 설정, 랜덤화 등.
"""

import asyncio
import colorsys
import itertools
import os
import shutil
import struct

import numpy as np
from PIL import Image
import omni.kit.asset_converter
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from sdg_config import *
from sdg_math import euler_to_rotation_matrix, build_camera_matrix
from sdg_usd_xform import (
    _distractor_prim_path_cache, _rep_prim_path_cache, _xformable_cache,
    _resolve_rep_prim_path,
)


# ============================================================
# 프로시저럴 텍스처 생성 (바닥/벽용)
# ============================================================
def _generate_procedural_textures(tex_dir, size=512):
    """시각적으로 확실히 구분되는 다양한 바닥/벽 텍스처 생성.

    7가지 패턴 × 각 3개 = 21장:
      - gray_concrete (회색 콘크리트)
      - green_epoxy (녹색 에폭시 바닥)
      - blue_epoxy (파란색 에폭시 바닥)
      - red_brick (적갈색 벽돌)
      - beige_tile (베이지 타일)
      - dark_asphalt (어두운 아스팔트)
      - wood_plank (목재 판자)
    """
    os.makedirs(tex_dir, exist_ok=True)
    # 항상 재생성 (이전 캐시된 회색 텍스처 무효화)
    existing = [f for f in os.listdir(tex_dir) if f.endswith(".png")]
    expected_count = 7 * 3
    # v10 마커 파일로 버전 체크
    marker = os.path.join(tex_dir, "_v10_marker")
    if os.path.exists(marker) and len(existing) >= expected_count:
        paths = [os.path.join(tex_dir, f) for f in sorted(existing) if f.endswith(".png")]
        print(f"  [TEX] Reusing {len(paths)} v10 procedural textures from {tex_dir}")
        return paths

    # 기존 파일 삭제 후 재생성
    for f in existing:
        os.remove(os.path.join(tex_dir, f))

    rng = np.random.RandomState(42)
    paths = []

    def _add_noise(img, rng, large_std=15, fine_std=8, size=512):
        """공통 노이즈 적용."""
        noise_l = rng.normal(0, large_std, (size // 8, size // 8))
        noise_l = np.repeat(np.repeat(noise_l, 8, axis=0), 8, axis=1)[:size, :size]
        noise_f = rng.normal(0, fine_std, (size, size))
        return img + noise_l + noise_f

    for idx in range(3):
        # 1) 회색 콘크리트
        base_rgb = np.array([140 + idx * 15, 138 + idx * 15, 135 + idx * 15], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 18, 10, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"gray_concrete_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 2) 녹색 에폭시 바닥 (창고에 흔함)
        g_base = rng.randint(100, 140)
        base_rgb = np.array([g_base * 0.45, g_base, g_base * 0.5], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 12, 6, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"green_epoxy_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 3) 파란색 에폭시 바닥
        b_base = rng.randint(110, 150)
        base_rgb = np.array([b_base * 0.4, b_base * 0.55, b_base], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 12, 6, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"blue_epoxy_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 4) 적갈색 벽돌 패턴
        brick_h = rng.choice([32, 48, 64])
        brick_w = brick_h * 2
        r_base, g_base_b, b_base_b = 160 + idx * 20, 85 + idx * 10, 60 + idx * 10
        img = np.full((size, size, 3), [r_base, g_base_b, b_base_b], dtype=np.float32)
        mortar = np.array([180, 175, 165], dtype=np.float32)
        for row in range(0, size, brick_h):
            img[row:min(row + 3, size), :, :] = mortar
            offset = (brick_w // 2) if ((row // brick_h) % 2) else 0
            for col in range(offset, size, brick_w):
                img[row:row + brick_h, max(col - 1, 0):min(col + 2, size), :] = mortar
        for c in range(3):
            img[:, :, c] += rng.normal(0, 8, (size, size))
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"red_brick_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 5) 베이지/크림 타일
        tile_sz = rng.choice([64, 128])
        t_base = np.array([210 - idx * 15, 195 - idx * 10, 170 - idx * 10], dtype=np.float32)
        img = np.full((size, size, 3), t_base, dtype=np.float32)
        grout = t_base * 0.6
        for i in range(0, size, tile_sz):
            img[max(i - 1, 0):i + 2, :, :] = grout
            img[:, max(i - 1, 0):i + 2, :] = grout
        for c in range(3):
            img[:, :, c] += rng.normal(0, 5, (size, size))
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"beige_tile_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 6) 어두운 아스팔트
        a_base = 65 + idx * 15
        base_rgb = np.array([a_base, a_base - 3, a_base - 5], dtype=np.float32)
        img = np.full((size, size, 3), base_rgb, dtype=np.float32)
        for c in range(3):
            img[:, :, c] = _add_noise(img[:, :, c], rng, 20, 12, size)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"dark_asphalt_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

        # 7) 목재 판자 (갈색 줄무늬)
        w_base = np.array([155 + idx * 15, 120 + idx * 10, 75 + idx * 8], dtype=np.float32)
        img = np.full((size, size, 3), w_base, dtype=np.float32)
        plank_w = rng.choice([48, 64, 96])
        for col in range(0, size, plank_w):
            plank_tint = rng.uniform(0.85, 1.15)
            img[:, col:col + plank_w, :] *= plank_tint
            img[:, max(col - 1, 0):col + 1, :] *= 0.6  # 판자 사이 어두운 선
        # 나뭇결 수평 줄무늬
        for _ in range(rng.randint(15, 30)):
            y = rng.randint(0, size)
            thickness = rng.randint(1, 4)
            img[y:y + thickness, :, :] *= rng.uniform(0.88, 0.96)
        for c in range(3):
            img[:, :, c] += rng.normal(0, 6, (size, size))
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = os.path.join(tex_dir, f"wood_plank_{idx:02d}.png")
        Image.fromarray(img).save(p); paths.append(p)

    # 버전 마커 생성
    with open(marker, "w") as f:
        f.write("v10")

    print(f"  [TEX] Generated {len(paths)} diverse procedural textures in {tex_dir}")
    return paths


def _classify_textures(texture_paths):
    """텍스처를 현실적(70%)/비현실적(30%) 카테고리로 분류하여 가중 샘플링용 데이터 반환.

    현실적: gray_concrete, green_epoxy, blue_epoxy, dark_asphalt (창고 바닥에 실제 존재)
    비현실적: red_brick, beige_tile, wood_plank (DR 다양성 목적, 빈도 낮게)
    """
    realistic = []   # 70% 확률
    stylized = []    # 30% 확률
    for p in texture_paths:
        basename = os.path.basename(p).lower()
        if any(k in basename for k in ("gray_concrete", "green_epoxy", "blue_epoxy", "dark_asphalt")):
            realistic.append(p)
        else:
            stylized.append(p)
    return realistic, stylized


def _pick_weighted_texture(rng, realistic, stylized, realistic_prob=0.7):
    """현실적 텍스처 70%, 비현실적 텍스처 30% 확률로 샘플링."""
    if not stylized or float(rng.random()) < realistic_prob:
        return realistic[int(rng.integers(len(realistic)))]
    else:
        return stylized[int(rng.integers(len(stylized)))]




# ============================================================
# USD 모델 바운딩박스 측정
# ============================================================
def compute_model_info(usd_path: str, target_size: float = PALLET_TARGET_SIZE):
    stage = Usd.Stage.Open(usd_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    rng = bbox.ComputeAlignedRange()
    mn = rng.GetMin()
    mx = rng.GetMax()
    size = mx - mn
    dims = [size[0], size[1], size[2]]
    longest = max(dims)
    scale = target_size / longest if longest > 0 else 1.0

    bbox_min = [float(mn[0]), float(mn[1]), float(mn[2])]
    bbox_max = [float(mx[0]), float(mx[1]), float(mx[2])]

    min_idx = dims.index(min(dims))
    # 가장 얇은 축이 Z가 되도록 회전
    if min_idx == 2:
        candidates = [(0, 0, 0)]
    elif min_idx == 1:
        candidates = [(90, 0, 0), (-90, 0, 0)]
    else:
        candidates = [(0, 90, 0), (0, -90, 0)]

    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ])

    # 메시 노멀 분석으로 윗면 방향 자동 판별
    # 팔레트 윗면은 슬롯/패턴이 많아 해당 방향의 노멀 수가 더 많음
    thin_axis = min_idx  # 가장 얇은 축 (0=X, 1=Y, 2=Z)
    plus_count = 0
    minus_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        normals = mesh.GetNormalsAttr().Get()
        if normals is None:
            continue
        for n in normals:
            val = n[thin_axis]
            if abs(val) > 0.5:  # 해당 축 방향으로 확실히 향하는 노멀만 카운트
                if val > 0:
                    plus_count += 1
                else:
                    minus_count += 1
    print(f"  {os.path.basename(usd_path)}: normal analysis axis={'XYZ'[thin_axis]}: "
          f"+{plus_count} / -{minus_count}")

    # 수동 오버라이드 테이블 (진단 렌더링으로 확인된 올바른 회전)
    # 노멀 분석은 팔레트처럼 상/하면 노멀 수가 비슷한 모델에서 불안정
    # Canonical base_rot: 모든 모델에서 X=medium, Y=long, Z=height(up)
    # edge 0→1=medium, edge 0→3=height, edge 0→4=long 으로 통일
    ORIENTATION_OVERRIDES = {
        "scene.usd": (180, 0, 90),     # Z-thin: Rz(90°)@Rx(180°) → Z반전으로 top/bottom 교정
        "scene_1.usd": (90, 0, 0),     # Y-thin: Rx(90°)로 Y(height)→Z, Z(long)→-Y
        "scene_2.usd": (90, 0, 0),     # Y-thin: 동일
        "scene_3.usd": (90, 0, 90),    # Y-thin: Rz(90°)@Rx(90°)로 X(long)→Y, Y(height)→Z
    }

    basename = os.path.basename(usd_path)
    if basename in ORIENTATION_OVERRIDES:
        base_rot = ORIENTATION_OVERRIDES[basename]
        print(f"  {basename}: using OVERRIDE rotation {base_rot}")
    else:
        # fallback: 노멀 분석 기반 자동 판별
        best_rot = candidates[0]
        if len(candidates) > 1:
            top_is_positive = plus_count >= minus_count
            for cand in candidates:
                R_test = euler_to_rotation_matrix(cand)
                axis_vec = np.zeros(3)
                axis_vec[thin_axis] = 1.0 if top_is_positive else -1.0
                rotated = R_test @ axis_vec
                if rotated[2] > 0:
                    best_rot = cand
                    break
            else:
                best_rot = candidates[0]
        base_rot = best_rot

    R_canonical = euler_to_rotation_matrix(base_rot)
    # Y↔Z swap: (X=med, Y=long, Z=height) → (X=med, Y=height, Z=long)
    # Rx(-90°): y→z, z→-y  ⇒  new_Y=old_Z(height), new_Z=-old_Y(long)
    R_yz_swap = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    R_canonical = R_yz_swap @ R_canonical
    rotated = (R_canonical @ corners.T).T

    # After R_yz_swap: Y=height(UP in world), Z=long(depth)
    # z_offset은 world Z(UP) 기준이므로 canonical Y(height)에서 계산
    min_y = rotated[:, 1].min()
    z_offset = -min_y * scale

    # Canonical bbox: base_rot 적용 후의 axis-aligned bbox
    canon_min = rotated.min(axis=0)
    canon_max = rotated.max(axis=0)
    canonical_bbox_min = [float(canon_min[0]), float(canon_min[1]), float(canon_min[2])]
    canonical_bbox_max = [float(canon_max[0]), float(canon_max[1]), float(canon_max[2])]

    print(f"  {os.path.basename(usd_path)}: dims=({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f}) "
          f"thin={'XYZ'[min_idx]} base_rot={base_rot} scale={scale:.6f} z_offset={z_offset:.4f}")
    print(f"    canonical bbox: min={canonical_bbox_min} max={canonical_bbox_max}")
    return scale, base_rot, bbox_min, bbox_max, z_offset, R_canonical, canonical_bbox_min, canonical_bbox_max


# ============================================================
# glTF -> USD 변환
# ============================================================
async def convert_gltf_to_usd(input_path: str, output_path: str) -> bool:
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = False
    converter_context.ignore_animations = True
    converter_context.ignore_cameras = True
    converter_context.single_mesh = False
    converter_context.smooth_normals = True
    converter_context.use_meter_as_world_unit = True

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        input_path, output_path,
        progress_callback=None,
        asset_converter_context=converter_context,
    )
    success = await task.wait_until_finished()
    if not success:
        status = task.get_status()
        error = task.get_detailed_error()
        print(f"[ERROR] convert fail {input_path}: {status} - {error}")
    return success


def _verify_textures(usd_path: str, gltf_dir: str):
    import shutil

    usd_dir = os.path.dirname(usd_path)
    tex_dir = os.path.join(usd_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    missing = []
    try:
        stage = Usd.Stage.Open(usd_path)
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            for inp_name in ("file", "filename", "inputs:file"):
                inp = shader.GetInput(inp_name)
                if not inp:
                    continue
                val = inp.Get()
                if val and hasattr(val, 'path') and val.path:
                    tex_ref = val.path
                    if not os.path.isabs(tex_ref):
                        abs_tex = os.path.normpath(os.path.join(usd_dir, tex_ref))
                    else:
                        abs_tex = tex_ref
                    if not os.path.exists(abs_tex):
                        missing.append((tex_ref, abs_tex))
    except Exception as e:
        print(f"[WARN] texture verification error: {e}")

    if not missing:
        print(f"  [TEX] {os.path.basename(usd_path)}: all textures OK")
        return

    search_dirs = [gltf_dir, os.path.join(gltf_dir, "textures"),
                   os.path.dirname(gltf_dir)]
    for tex_ref, abs_tex in missing:
        tex_name = os.path.basename(tex_ref)
        found = False
        for sdir in search_dirs:
            candidate = os.path.join(sdir, tex_name)
            if os.path.exists(candidate):
                dest = abs_tex
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(candidate, dest)
                print(f"  [TEX] copied: {candidate} -> {dest}")
                found = True
                break
        if not found:
            print(f"  [TEX] WARNING: texture missing! {tex_ref} (searched: {search_dirs})")


def convert_all_gltf(gltf_dir: str, gltf_files: list, usd_dir: str) -> list:
    os.makedirs(usd_dir, exist_ok=True)
    usd_paths = []

    loop = asyncio.get_event_loop()
    for gltf_name in gltf_files:
        usd_name = os.path.splitext(gltf_name)[0] + ".usd"
        usd_path = os.path.join(usd_dir, usd_name)

        if os.path.exists(usd_path):
            print(f"[INFO] USD cache: {usd_path}")
            usd_paths.append(usd_path)
            _verify_textures(usd_path, gltf_dir)
            continue

        gltf_path = os.path.join(gltf_dir, gltf_name)
        if not os.path.exists(gltf_path):
            print(f"[WARN] glTF not found and no USD cache: {gltf_path}")
            continue

        print(f"[INFO] converting: {gltf_path} -> {usd_path}")
        success = loop.run_until_complete(convert_gltf_to_usd(gltf_path, usd_path))
        if success:
            usd_paths.append(usd_path)
            print(f"[OK] done: {usd_path}")
            _verify_textures(usd_path, gltf_dir)
        else:
            print(f"[ERROR] failed: {gltf_path}")

    return usd_paths


# ============================================================
# 팔레트 색상 선택 (60% 프리셋, 40% HSV 연속 랜덤)
# ============================================================
def _pick_pallet_color(rng):
    if rng.random() < 0.6:
        return PALLET_COLORS[rng.integers(len(PALLET_COLORS))]
    else:
        h = float(rng.uniform(0.0, 1.0))
        s = float(rng.uniform(0.2, 1.0))
        v = float(rng.uniform(0.25, 0.85))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b)


# ============================================================
# v4: warehouse 환경 로드 (4종 변형 중 랜덤 선택)
# ============================================================
def _try_load_warehouse(stage):
    """warehouse 변형을 로컬 에셋에서 랜덤 선택하여 로드. 성공 여부 반환."""
    import random as _rnd

    # 로컬 에셋에서 사용 가능한 변형 찾기
    available_variants = [p for p in WAREHOUSE_VARIANTS_LOCAL if os.path.isfile(p)]
    if available_variants:
        _rnd.shuffle(available_variants)

    # 시도할 경로 목록: 로컬 변형들 우선, Nucleus fallback
    paths_to_try = [("local", p) for p in available_variants]
    paths_to_try.append(("Nucleus", WAREHOUSE_USD))

    for source, path in paths_to_try:
        try:
            from pxr import UsdGeom as _UG
            warehouse_prim = stage.DefinePrim("/World/Warehouse", "Xform")
            warehouse_prim.GetReferences().AddReference(path)

            children = list(warehouse_prim.GetChildren())
            if children:
                variant_name = os.path.basename(path)
                print(f"  [ENV] {variant_name} loaded from {source}: {path}")
                # 창고 내부 팔레트 숨기기 — 타겟 팔레트와 혼동 방지
                from pxr import UsdGeom as _UG_wh
                hidden_count = 0
                for desc in Usd.PrimRange(warehouse_prim):
                    desc_name = desc.GetName().lower()
                    if "pallet" in desc_name or "palette" in desc_name:
                        _UG_wh.Imageable(desc).MakeInvisible()
                        hidden_count += 1
                if hidden_count:
                    print(f"  [ENV] Hidden {hidden_count} warehouse pallet prims")
                return True
            else:
                warehouse_prim.GetReferences().ClearReferences()
                stage.RemovePrim(warehouse_prim.GetPath())
                print(f"  [ENV] {os.path.basename(path)} not resolved from {source}, trying next...")
        except Exception as e:
            print(f"  [ENV] warehouse load failed ({source}): {e}")

    print(f"  [ENV] warehouse not available, falling back to floor/wall")
    return False


def _list_nucleus_dir(url, max_depth=0, _depth=0):
    """omni.client로 Nucleus/S3 디렉토리를 열거. USD 파일 경로 리스트 반환."""
    results = []
    try:
        import omni.client
        result, entries = omni.client.list(url)
        if result != omni.client.Result.OK:
            return results
        for entry in entries:
            full = f"{url}/{entry.relative_path}"
            if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                if _depth < max_depth:
                    results.extend(_list_nucleus_dir(full, max_depth, _depth + 1))
            elif full.endswith(".usd") or full.endswith(".usda") or full.endswith(".usdz"):
                results.append(full)
    except Exception as e:
        print(f"  [LIST] omni.client.list failed for {url}: {e}")
    return results


def _try_load_props(stage):
    """Isaac Sim Props를 로컬 에셋 → Nucleus 순서로 로드 시도. 성공한 경로 리스트 반환."""
    import omni.replicator.core as rep

    # 1. 로컬 에셋 우선 확인 (Nucleus 불필요) — 카테고리별 검증
    local_loaded = []
    local_by_category = {}
    if os.path.isdir(ISAAC_ASSETS_ROOT):
        print(f"  [PROPS] Local assets root: {ISAAC_ASSETS_ROOT}")
        for cat_name, cat_props in DISTRACTOR_CATEGORIES.items():
            cat_loaded = []
            for rel_path in cat_props:
                full_path = os.path.join(ISAAC_ASSETS_ROOT, rel_path).replace("\\", "/")
                if os.path.isfile(full_path):
                    cat_loaded.append(full_path)
                    local_loaded.append(full_path)
            local_by_category[cat_name] = cat_loaded
            print(f"    [{cat_name}] {len(cat_loaded)}/{len(cat_props)} available")

    if local_loaded:
        print(f"  [PROPS] {len(local_loaded)} props total from local assets")
        return local_loaded

    # 2. Nucleus fallback
    print(f"  [PROPS] No local assets, trying Nucleus...")
    assets_root = None
    try:
        from omni.isaac.nucleus import get_assets_root_path
        assets_root = get_assets_root_path()
        if assets_root:
            print(f"  [PROPS] Assets root: {assets_root}")
    except (ImportError, Exception) as e:
        print(f"  [PROPS] get_assets_root_path failed: {e}")

    candidates = []
    if assets_root:
        for rel in DISTRACTOR_PROPS_LOCAL:
            candidates.append(f"{assets_root}/{rel}")
    candidates.extend(DISTRACTOR_PROPS)

    loaded = []
    tried_paths = set()
    for prop_path in candidates:
        if prop_path in tried_paths:
            continue
        tried_paths.add(prop_path)
        try:
            test_prim = rep.create.from_usd(prop_path, count=1)
            if test_prim is not None:
                loaded.append(prop_path)
                print(f"  [PROPS] OK: {prop_path}")
                with test_prim:
                    rep.modify.pose(position=(0, 0, -200))
            else:
                print(f"  [PROPS] FAIL (None): {prop_path}")
        except Exception as e:
            err_short = str(e)[:80]
            print(f"  [PROPS] FAIL: {prop_path} -> {err_short}")

    if loaded:
        print(f"  [PROPS] {len(loaded)} props available from Nucleus")
    else:
        print(f"  [PROPS] No props available, using primitive fallback")
    return loaded


# ============================================================
# 씬 구성
# ============================================================
def setup_scene(usd_paths: list, model_infos: list,
                image_width: int = IMAGE_WIDTH, image_height: int = IMAGE_HEIGHT,
                hdri_dir: str = None, chosen_renderer: str = "RayTracedLighting",
                settings=None):
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    rt_subframes = RT_SUBFRAMES_PT if chosen_renderer == "PathTracing" else RT_SUBFRAMES_RTL

    # === v5: 배경 혼합 — 창고 + 프로시저럴 바닥/벽 둘 다 생성, 매 프레임 토글 ===
    warehouse_loaded = _try_load_warehouse(stage)

    # 프로시저럴 바닥/벽 — USD API로 머티리얼 직접 생성 (Replicator 그래프 충돌 방지)
    print(f"  [ENV] Creating procedural floor/walls (always available)")

    # 바닥/벽 머티리얼: Replicator로 생성 (MDL 속성 자동 설정)
    # rep.distribution.*를 사용하지 않으므로 Replicator 그래프가 매 step마다 변경하지 않음
    # 텍스처 변경은 USD API (_change_floor_wall_textures)로만 수행
    _floor_mat = rep.create.material_omnipbr(
        diffuse=(1.0, 1.0, 1.0),  # white tint — 텍스처 원본 색상 유지
        roughness=0.6,
        metallic=0.0,
        count=1,
    )
    _wall_mat = rep.create.material_omnipbr(
        diffuse=(1.0, 1.0, 1.0),
        roughness=0.6,
        metallic=0.0,
        count=1,
    )

    _floor_plane = rep.create.plane(
        scale=20,
        position=(0, 0, -0.001),
        visible=not warehouse_loaded,
        semantics=[("class", "floor")],
        material=_floor_mat,
    )

    _wall_planes = []
    for wall_pos, wall_rot in [
        ((0, 10, 5), (-90, 0, 0)),
        ((0, -10, 5), (90, 0, 0)),
        ((-10, 0, 5), (0, -90, 0)),
        ((10, 0, 5), (0, 90, 0)),
    ]:
        _wall_planes.append(rep.create.plane(
            scale=20,
            position=wall_pos,
            rotation=wall_rot,
            visible=not warehouse_loaded,
            semantics=[("class", "wall")],
            material=_wall_mat,
        ))
    # 천장
    _wall_planes.append(rep.create.plane(
        scale=20,
        position=(0, 0, 10),
        rotation=(180, 0, 0),
        visible=not warehouse_loaded,
        semantics=[("class", "wall")],
        material=_wall_mat,
    ))

    # === v3: HDRI dome light (배경 보강) ===
    hdri_files = []
    # 명시적 hdri_dir → 로컬 에셋 HDRI 순으로 탐색
    hdri_search_dirs = []
    if hdri_dir and os.path.isdir(hdri_dir):
        hdri_search_dirs.append(hdri_dir)
    if os.path.isdir(HDRI_DIR_LOCAL):
        for sub in ("Indoor", "Clear"):
            sub_dir = os.path.join(HDRI_DIR_LOCAL, sub)
            if os.path.isdir(sub_dir):
                hdri_search_dirs.append(sub_dir)
    for hdir in hdri_search_dirs:
        for f in os.listdir(hdir):
            if f.lower().endswith((".hdr", ".exr")):
                hdri_files.append(os.path.join(hdir, f).replace("\\", "/"))
    if hdri_files:
        print(f"  [HDRI] {len(hdri_files)} HDRI files found")

    # === DomeLight ===
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(3000)
    dome_light.CreateColorAttr(Gf.Vec3f(0.90, 0.88, 0.85))
    dome_light.CreateTextureFormatAttr("automatic")
    if hdri_files:
        dome_light.CreateTextureFileAttr(hdri_files[0])
        print(f"  [ENV] DomeLight created with HDRI ({len(hdri_files)} files, will randomize)")
    else:
        print(f"  [ENV] DomeLight created (no HDRI, color-based ambient)")

    # === v3: 조명 - RectLight 2~3개 (천장 형광등 시뮬레이션) ===
    # 주 조명: 천장 형광등 (항상 ON)
    main_light = rep.create.light(
        light_type="Rect",
        position=(0, 0, 5),
        rotation=(-90, 0, 0),           # 아래를 비춤
        intensity=100000,
        color=(0.95, 0.92, 0.88),
        scale=(3.0, 1.5, 1.0),          # 형광등 형태: 가로로 긴 직사각형
        count=1,
    )

    # 보조 조명 1: 측면 (창문/출입구 빛)
    fill_light_1 = rep.create.light(
        light_type="Rect",
        position=(3, -3, 4),
        rotation=(-70, 30, 0),
        intensity=50000,
        color=(0.90, 0.90, 0.95),
        scale=(2.0, 1.0, 1.0),
        count=1,
    )

    # 보조 조명 2: 반대편 측면
    fill_light_2 = rep.create.light(
        light_type="Rect",
        position=(-3, 2, 4),
        rotation=(-60, -20, 0),
        intensity=40000,
        color=(0.85, 0.85, 0.80),
        scale=(2.0, 1.0, 1.0),
        count=1,
    )

    scene_lights = [main_light, fill_light_1, fill_light_2]

    # === v5: 디스트랙터 - 카테고리별 균등 샘플링으로 다양성 확보 ===
    available_props = _try_load_props(stage)
    use_props = len(available_props) > 0

    distractor_prims = []
    if use_props:
        # v5: 카테고리별 균등 풀 생성 — MAX_DISTRACTOR_POOL개 미리 생성
        # 매 프레임 이 중 MAX_DISTRACTORS_PER_FRAME개를 랜덤 선택하여 표시
        cat_available = {}
        for cat_name in DISTRACTOR_CATEGORIES:
            cat_paths = []
            for rel in DISTRACTOR_CATEGORIES[cat_name]:
                fp = os.path.join(ISAAC_ASSETS_ROOT, rel).replace("\\", "/")
                if fp in available_props:
                    cat_paths.append(fp)
            if cat_paths:
                cat_available[cat_name] = cat_paths

        # 각 카테고리에서 라운드로빈으로 풀 크기만큼 선택
        import itertools
        cat_cycle = itertools.cycle(list(cat_available.keys()))
        cat_indices = {c: 0 for c in cat_available}
        selected_props = []
        for _ in range(MAX_DISTRACTOR_POOL):
            cat = next(cat_cycle)
            paths = cat_available[cat]
            selected_props.append((cat, paths[cat_indices[cat] % len(paths)]))
            cat_indices[cat] += 1

        for i, (cat, prop_path) in enumerate(selected_props):
            d = rep.create.from_usd(
                prop_path,
                semantics=[("class", "distractor")],
                count=1,
            )
            with d:
                rep.modify.pose(position=(0, 0, -200))
            distractor_prims.append(d)
        cat_summary = {}
        for cat, _ in selected_props:
            cat_summary[cat] = cat_summary.get(cat, 0) + 1
        print(f"  [DIST] Created {len(selected_props)} distractor pool (category-balanced):")
        for cat, cnt in cat_summary.items():
            print(f"    [{cat}] x{cnt}")
    else:
        # Enhanced fallback: 다양한 primitive로 창고 물체 시뮬레이션
        _pool = MAX_DISTRACTOR_POOL
        n_cube = max(1, int(_pool * 0.4))
        n_cyl = max(1, int(_pool * 0.25))
        n_cone = max(1, int(_pool * 0.2))
        n_sphere = _pool - n_cube - n_cyl - n_cone
        distractor_types = (["cube"] * n_cube + ["cylinder"] * n_cyl +
                            ["cone"] * n_cone + ["sphere"] * max(0, n_sphere))
        distractor_types = distractor_types[:_pool]

        for i in range(len(distractor_types)):
            dtype = distractor_types[i]
            create_fn = {
                "cube": rep.create.cube,
                "cylinder": rep.create.cylinder,
                "cone": rep.create.cone,
                "sphere": rep.create.sphere,
            }[dtype]
            d = create_fn(
                semantics=[("class", "distractor")],
                position=(0, 0, -200),
                scale=0.3,
                visible=False,
            )
            distractor_prims.append(d)
        type_counts = {t: distractor_types.count(t) for t in set(distractor_types)}
        print(f"  [DIST] Created {len(distractor_types)} distractor pool (primitive fallback: {type_counts})")

    # === v8: 디스트랙터 bbox 측정 — USD 파일에서 직접 읽기 ===
    _distractor_sizes = []  # 각 distractor의 원본 max dimension (m)
    if use_props:
        # USD prop 파일을 직접 열어서 bbox 측정 (Replicator node 접근 불필요)
        _prop_bbox_cache = {}  # path -> max_dim 캐시
        for _cat, _prop_path in selected_props:
            if _prop_path in _prop_bbox_cache:
                _distractor_sizes.append(_prop_bbox_cache[_prop_path])
                continue
            _max_dim = 0.5  # fallback
            try:
                _prop_stage = Usd.Stage.Open(_prop_path)
                if _prop_stage:
                    _bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    _root = _prop_stage.GetPseudoRoot()
                    _extent = _bc.ComputeWorldBound(_root).GetRange()
                    _sz = _extent.GetSize()
                    _max_dim = max(float(_sz[0]), float(_sz[1]), float(_sz[2]))
                    if _max_dim <= 0:
                        _max_dim = 0.5
            except Exception as _e:
                print(f"    [DIST] bbox fail for {os.path.basename(_prop_path)}: {_e}")
            _prop_bbox_cache[_prop_path] = _max_dim
            _distractor_sizes.append(_max_dim)
    else:
        # Primitive fallback: 기본 scale=0.3으로 생성됨 → unit prim(1m) * 0.3 = 0.3m
        for _ in distractor_prims:
            _distractor_sizes.append(0.3)
    if _distractor_sizes:
        print(f"  [DIST] Bbox measured: min={min(_distractor_sizes):.3f}m, "
              f"max={max(_distractor_sizes):.3f}m, mean={sum(_distractor_sizes)/len(_distractor_sizes):.3f}m")

    # === 머티리얼 풀 ===
    # 프로시저럴 텍스처 생성 (콘크리트/에폭시/타일/아스팔트)
    proc_textures = _generate_procedural_textures(PROCEDURAL_TEX_DIR)
    # 텍스처 경로를 정방향 슬래시로 변환 (OmniPBR 호환)
    proc_textures = [p.replace("\\", "/") for p in proc_textures]

    # 바닥/벽 머티리얼: rep.create.material_omnipbr()를 사용하지 않음.
    # rep.create.plane()이 생성하는 기본 OmniPBR 셰이더에 직접 diffuse_texture
    # 입력을 추가하고, 매 프레임 _change_floor_wall_textures()로 텍스처 경로를 변경.
    # 이렇게 하면 Replicator 그래프와의 충돌(무지개 모자이크)이 완전히 방지됨.
    tex_real, tex_style = _classify_textures(proc_textures)
    print(f"  [MAT] Floor/wall textures: {len(tex_real)} realistic (70%) + {len(tex_style)} stylized (30%) - USD API only")
    pallet_materials = rep.create.material_omnipbr(
        diffuse=(0.5, 0.5, 0.5),
        roughness=0.6,
        metallic=0.05,
        count=len(PALLET_COLORS),
    )

    # v10: 디스트랙터 머티리얼 — 각 디스트랙터에 1개씩 미리 생성 + shader 캐싱
    # 매 프레임 rep.create.material_omnipbr()를 호출하면 Replicator 그래프 노드가
    # 무한히 누적되어 프레임이 진행될수록 기하급수적으로 느려짐.
    # 초기화 시 1회만 생성하고, 매 프레임 USD API로 파라미터만 변경.
    _distractor_shader_cache = []  # 각 디스트랙터의 OmniPBR shader prim
    for _di, _dp in enumerate(distractor_prims):
        _dmat = rep.create.material_omnipbr(
            diffuse=(0.5, 0.5, 0.5),
            roughness=0.5,
            metallic=0.05,
            count=1,
        )
        with _dp:
            rep.randomizer.materials(_dmat)
        _distractor_shader_cache.append(None)  # shader prim은 step() 후 resolve

    # shader prim resolve를 위해 1 step 실행
    rep.orchestrator.step(rt_subframes=1)
    rep.orchestrator.wait_until_complete()

    # v10: shader 캐시 resolve — 디스트랙터 prim을 stage에서 직접 탐색
    _stage_tmp = omni.usd.get_context().get_stage()
    from pxr import UsdShade

    # 방법 1: Replicator 객체에서 prim path 추출 시도 (여러 방법)
    for _di, _dp in enumerate(distractor_prims):
        try:
            _dp_path = None
            # Replicator NodeType object의 prim path 추출
            if hasattr(_dp, 'node'):
                try:
                    _dp_path = _dp.node.get_prim_path()
                except Exception:
                    pass
            if not _dp_path and hasattr(_dp, 'get_output_prims'):
                try:
                    _out = _dp.get_output_prims()
                    if _out and len(_out) > 0:
                        _dp_path = str(_out[0].GetPath()) if hasattr(_out[0], 'GetPath') else str(_out[0])
                except Exception:
                    pass
            if not _dp_path:
                continue

            _dp_prim = _stage_tmp.GetPrimAtPath(_dp_path)
            if not _dp_prim or not _dp_prim.IsValid():
                continue

            # material binding 탐색 (자신 + 하위 mesh 포함)
            _found = False
            _prims_to_check = [_dp_prim] + list(_dp_prim.GetAllChildren())
            for _check_prim in _prims_to_check:
                _bind_api = UsdShade.MaterialBindingAPI(_check_prim)
                _bound = _bind_api.GetDirectBinding()
                _mat_path = _bound.GetMaterialPath()
                if _mat_path:
                    _mat_prim = _stage_tmp.GetPrimAtPath(_mat_path)
                    if _mat_prim and _mat_prim.IsValid():
                        for _child in _mat_prim.GetAllChildren():
                            if _child.GetTypeName() == "Shader":
                                _distractor_shader_cache[_di] = _child
                                _found = True
                                break
                if _found:
                    break
        except Exception as _e:
            print(f"    [MAT] shader cache #{_di} error: {_e}")

    # 방법 2: 캐시 실패 시, /Replicator/ 하위 모든 Shader를 수집하여 순서대로 할당
    _cached_count = sum(1 for s in _distractor_shader_cache if s is not None)
    if _cached_count < len(distractor_prims):
        # 디스트랙터용 머티리얼은 마지막 N개의 OmniPBR Shader
        _all_shaders = []
        for _p in _stage_tmp.Traverse():
            _pp = str(_p.GetPath())
            if "/Replicator/Looks/" in _pp and _p.GetTypeName() == "Shader":
                _all_shaders.append(_p)

        # 디스트랙터 머티리얼은 setup 순서대로 생성되었으므로,
        # 마지막 len(distractor_prims)개가 디스트랙터용 머티리얼
        # (앞쪽은 floor, wall, pallet 머티리얼)
        _n_dist = len(distractor_prims)
        if len(_all_shaders) >= _n_dist:
            _dist_shaders = _all_shaders[-_n_dist:]
            for _di in range(len(distractor_prims)):
                if _distractor_shader_cache[_di] is None:
                    _distractor_shader_cache[_di] = _dist_shaders[_di]
            print(f"    [MAT] Fallback: assigned {_n_dist} shaders from /Replicator/Looks/ (total {len(_all_shaders)} shaders)")

    _cached_count = sum(1 for s in _distractor_shader_cache if s is not None)
    print(f"  [MAT] Distractor shader cache: {_cached_count}/{len(distractor_prims)} resolved")

    # 팔레트 로드
    pallet_prims = []
    for i, usd_path in enumerate(usd_paths):
        usd_uri = usd_path.replace("\\", "/")
        if not usd_uri.startswith("/"):
            usd_uri = "/" + usd_uri

        s, base_rot, bbox_min, bbox_max, z_offset, R_canonical, cbbox_min, cbbox_max = model_infos[i]
        pallet = rep.create.from_usd(
            usd_uri,
            semantics=[("class", "pallet")],
            count=1,
        )
        # 초기 배치: rep.modify.pose()로 1회 설정 (setup 시만, 이후 USD API로 제어)
        with pallet:
            rep.modify.pose(
                position=(i * 50, 0, z_offset),
                rotation=base_rot,
                scale=(s, s, s),
            )
        pallet_prims.append((pallet, s, base_rot, bbox_min, bbox_max, z_offset, R_canonical, cbbox_min, cbbox_max))

    # 카메라
    camera = rep.create.camera(
        position=(0, -3, 1.0),
        look_at=(0, 0, 0),
        focal_length=FOCAL_LENGTH,
        horizontal_aperture=SENSOR_WIDTH,
        clipping_range=(0.01, 100.0),
    )
    render_product = rep.create.render_product(camera, (image_width, image_height))

    # K 행렬 계산
    actual_focal = FOCAL_LENGTH
    actual_aperture = SENSOR_WIDTH
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            actual_aperture = cam.GetHorizontalApertureAttr().Get()
            actual_focal = cam.GetFocalLengthAttr().Get()
            print(f"  camera: focal={actual_focal}mm, h_aperture={actual_aperture}mm")
            break

    K = build_camera_matrix(actual_focal, actual_aperture, image_width, image_height)
    print(f"  K: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    # prim path 수집
    pallet_prim_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        parts = path.split("/")
        if len(parts) == 3 and "Ref_Xform" in parts[2]:
            pallet_prim_paths.append(path)
    pallet_prim_paths.sort()
    print(f"  pallet prim paths: {pallet_prim_paths}", flush=True)

    # 렌더러 모드 재확인
    if settings is not None:
        _mode_after_scene = settings.get("/rtx/rendermode")
        print(f"  [RENDER] post-scene rendermode = {_mode_after_scene}")
        if _mode_after_scene != chosen_renderer:
            print(f"  [RENDER] rendermode reset after scene load! Re-applying {chosen_renderer}...")
            settings.set("/rtx/rendermode", chosen_renderer)

        # DLSS/AA 재비활성화
        settings.set("/rtx/post/dlss/enabled", False)
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/post/aa/enabled", False)
        settings.set("/rtx/renderScaleFactor", 1.0)
        settings.set("/rtx/upscaler/enabled", False)

    # warm-up
    for _ in range(WARMUP_STEPS):
        rep.orchestrator.step(rt_subframes=rt_subframes)
        rep.orchestrator.wait_until_complete()
    print(f"  warm-up done ({WARMUP_STEPS} steps, rt_subframes={rt_subframes})", flush=True)

    # v11: warm-up 후 prim path resolve (USD stage 스캔 fallback 포함)
    _stage_tmp = omni.usd.get_context().get_stage()

    # 1) 디스트랙터 prim path resolve
    _resolved_count = 0
    for _di, _dp in enumerate(distractor_prims):
        _dpath = _resolve_rep_prim_path(_dp)
        if _dpath:
            _resolved_count += 1
    print(f"  [v11] Distractor prim path direct resolve: {_resolved_count}/{len(distractor_prims)}")

    # Fallback: stage 스캔으로 semantic:class=distractor 인 prim 찾기
    if _resolved_count < len(distractor_prims):
        _dist_xform_paths = []
        for _p in _stage_tmp.Traverse():
            _pp = str(_p.GetPath())
            if not _pp.startswith("/Replicator/"):
                continue
            if "/Looks/" in _pp or "Plane" in _pp:
                continue
            # Xform 타입의 직계 자식만 (중첩 mesh 제외)
            if _p.GetParent() and str(_p.GetParent().GetPath()) == "/Replicator":
                # semantic class 확인
                _sem_attr = _p.GetAttribute("semantic:Semantics:params:semanticData")
                if _sem_attr and _sem_attr.IsValid():
                    _sem_val = _sem_attr.Get()
                    if _sem_val == "distractor":
                        _dist_xform_paths.append(_pp)

        print(f"  [v11] Stage scan found {len(_dist_xform_paths)} distractor prims")
        if len(_dist_xform_paths) >= len(distractor_prims):
            for _di, _dp in enumerate(distractor_prims):
                _existing = _resolve_rep_prim_path(_dp)
                if not _existing:
                    _rep_prim_path_cache[id(_dp)] = _dist_xform_paths[_di]
                    _distractor_prim_path_cache[id(_dp)] = _dist_xform_paths[_di]
            _resolved_count = sum(1 for _dp in distractor_prims if _resolve_rep_prim_path(_dp))
            print(f"  [v11] After fallback: {_resolved_count}/{len(distractor_prims)} resolved")
            if _resolved_count > 0:
                print(f"  [v11] Sample: #{0}={_resolve_rep_prim_path(distractor_prims[0])}")

    # 2) 팔레트 prim path resolve
    _pallet_resolved = 0
    for _pi, (_pp_prim, *_rest) in enumerate(pallet_prims):
        _ppath = _resolve_rep_prim_path(_pp_prim)
        if _ppath:
            _pallet_resolved += 1

    if _pallet_resolved < len(pallet_prims):
        # Fallback: semantic:class=pallet 인 prim 찾기
        _pallet_xform_paths = []
        for _p in _stage_tmp.Traverse():
            _pp = str(_p.GetPath())
            if not _pp.startswith("/Replicator/"):
                continue
            if _p.GetParent() and str(_p.GetParent().GetPath()) == "/Replicator":
                _sem_attr = _p.GetAttribute("semantic:Semantics:params:semanticData")
                if _sem_attr and _sem_attr.IsValid() and _sem_attr.Get() == "pallet":
                    _pallet_xform_paths.append(_pp)
        if len(_pallet_xform_paths) >= len(pallet_prims):
            for _pi, (_pp_prim, *_rest) in enumerate(pallet_prims):
                if not _resolve_rep_prim_path(_pp_prim):
                    _rep_prim_path_cache[id(_pp_prim)] = _pallet_xform_paths[_pi]
            _pallet_resolved = sum(1 for (_pp_prim, *_) in pallet_prims if _resolve_rep_prim_path(_pp_prim))

    print(f"  [v11] Pallet prim path resolved: {_pallet_resolved}/{len(pallet_prims)}")

    # v11: resolve된 경로가 그래프 노드 경로인 경우, 실제 USD prim을 찾아야 함
    # get_output_prims()가 반환하는 것은 SDGPipeline 경로이므로,
    # 실제 렌더링되는 prim을 찾기 위해 다른 방법 사용
    _need_scene_scan = False
    for _di in range(min(2, len(distractor_prims))):
        _dpath = _resolve_rep_prim_path(distractor_prims[_di])
        if _dpath:
            _dprim = _stage_tmp.GetPrimAtPath(_dpath)
            _has_xform = False
            if _dprim and _dprim.IsValid():
                _xf = UsdGeom.Xformable(_dprim)
                _has_xform = len(_xf.GetOrderedXformOps()) > 0
            if not _has_xform:
                _need_scene_scan = True
                print(f"  [v11] dist#{_di} path={_dpath} has NO xformOps -> need scene scan")
                break

    if _need_scene_scan:
        # Replicator 그래프 노드가 아닌 실제 씬 prim 찾기
        # semantic:Semantics:params:semanticData 기준으로 탐색
        print(f"  [v11] Scanning stage for actual scene prims...")
        # /Replicator/ 직계 자식 중 xformOps가 있는 Xform prim 나열
        _replicator_prim = _stage_tmp.GetPrimAtPath("/Replicator")
        _xform_children = []
        if _replicator_prim and _replicator_prim.IsValid():
            for _child in _replicator_prim.GetChildren():
                _cp = str(_child.GetPath())
                if "Looks" in _cp or "SDGPipeline" in _cp:
                    continue
                _xf = UsdGeom.Xformable(_child)
                _ops = _xf.GetOrderedXformOps()
                if len(_ops) > 0:
                    # translate z 값으로 분류
                    _tz = None
                    for _op in _ops:
                        if "translate" in str(_op.GetName()):
                            _tv = _op.Get()
                            if _tv:
                                _tz = float(_tv[2]) if hasattr(_tv, '__getitem__') else None
                            break
                    _xform_children.append((_cp, _tz, len(_ops)))

        print(f"  [v11] /Replicator/ xform children: {len(_xform_children)}")
        for _i, (_path, _tz, _nops) in enumerate(_xform_children[:40]):
            _label = "DIST?" if _tz is not None and _tz < -150 else ("PAL?" if _tz is not None and _tz < -50 else "OTHER")
            print(f"    [{_i}] {_path} tz={_tz} nops={_nops} -> {_label}")

        # z=-200 인 것 = 디스트랙터, z=-100 인 것 = 팔레트 (초기 숨김 위치)
        _dist_scene_paths = [p for p, tz, _ in _xform_children if tz is not None and tz < -150]
        _pallet_scene_paths = [p for p, tz, _ in _xform_children if tz is not None and -150 < tz < -50]
        # z가 0 근처인 것도 팔레트일 수 있음 (warm-up에서 활성화된 팔레트)
        _pallet_scene_paths += [p for p, tz, _ in _xform_children
                                if tz is not None and -50 <= tz <= 5
                                and p not in _dist_scene_paths
                                and "Plane" not in p]

        print(f"  [v11] Classified: {len(_dist_scene_paths)} distractors (z<-150), {len(_pallet_scene_paths)} pallets (z~-100 or z~0)")

        # 디스트랙터 매핑
        if len(_dist_scene_paths) >= len(distractor_prims):
            for _di, _dp in enumerate(distractor_prims):
                _rep_prim_path_cache[id(_dp)] = _dist_scene_paths[_di]
                _distractor_prim_path_cache[id(_dp)] = _dist_scene_paths[_di]
            _xformable_cache.clear()
            print(f"  [v11] Distractor paths remapped: {len(distractor_prims)} prims")
            print(f"  [v11] Sample: dist#0 = {_dist_scene_paths[0]}")

        # 팔레트 매핑
        if len(_pallet_scene_paths) >= len(pallet_prims):
            for _pi, (_pp_prim, *_rest) in enumerate(pallet_prims):
                _rep_prim_path_cache[id(_pp_prim)] = _pallet_scene_paths[_pi]
            _xformable_cache.clear()
            print(f"  [v11] Pallet paths remapped: {len(pallet_prims)} prims")
            print(f"  [v11] Sample: pallet#0 = {_pallet_scene_paths[0]}")

    prebuilt = {
        "lights": scene_lights,
        "pallet_materials": pallet_materials,
        "distractor_prims": distractor_prims,
        "distractor_sizes": _distractor_sizes,
        "distractor_shader_cache": _distractor_shader_cache,
        "hdri_files": hdri_files,
        "warehouse_loaded": warehouse_loaded,
        "use_props": use_props,
        "dome_light": dome_light,
        "proc_textures": proc_textures,
        "tex_realistic": _classify_textures(proc_textures)[0],
        "tex_stylized": _classify_textures(proc_textures)[1],
        "usd_paths": usd_paths,
    }

    return pallet_prims, camera, render_product, pallet_prim_paths, K, prebuilt


# ============================================================
# v3: Domain Randomization
# ============================================================
def register_randomizers(pallet_prims, camera, prebuilt):
    """v12: 조명 randomizer만 on_frame 트리거로 유지.
    바닥/벽은 USD API로, 팔레트/디스트랙터 pose는 generate_data 루프에서 처리.
    """
    _main_light, _fill_light_1, _fill_light_2 = prebuilt["lights"]
    _dome_light = prebuilt["dome_light"]
    _hdri_files = prebuilt.get("hdri_files", [])

    def randomize_lights():
        # DomeLight
        if _dome_light is not None:
            _dome_intensity = float(np.random.uniform(2000, 3500))
            _dome_light.GetIntensityAttr().Set(_dome_intensity)
            if _hdri_files:
                _hdri_path = str(np.random.choice(_hdri_files))
                _dome_light.GetTextureFileAttr().Set(_hdri_path)

        with _main_light:
            rep.modify.attribute(
                "color",
                rep.distribution.uniform((0.85, 0.82, 0.78), (1.0, 0.98, 0.95)),
            )
            rep.modify.attribute(
                "intensity",
                rep.distribution.uniform(100000, 300000),
            )
            rep.modify.pose(
                position=rep.distribution.uniform((-3, -3, 4), (3, 3, 6)),
                rotation=rep.distribution.uniform((-100, -20, -20), (-70, 20, 20)),
            )
        with _fill_light_1:
            rep.modify.attribute(
                "intensity",
                rep.distribution.uniform(50000, 200000),
            )
            rep.modify.attribute(
                "color",
                rep.distribution.uniform((0.7, 0.7, 0.75), (1.0, 1.0, 0.95)),
            )
            rep.modify.pose(
                position=rep.distribution.uniform((-5, -5, 2), (5, 5, 5)),
            )
            rep.modify.visibility(
                rep.distribution.choice([True, True, True, False])
            )
        with _fill_light_2:
            rep.modify.attribute(
                "intensity",
                rep.distribution.uniform(40000, 180000),
            )
            rep.modify.attribute(
                "color",
                rep.distribution.uniform((0.75, 0.75, 0.75), (1.0, 1.0, 0.95)),
            )
            rep.modify.pose(
                position=rep.distribution.uniform((-5, -5, 2), (5, 5, 5)),
            )
            rep.modify.visibility(
                rep.distribution.choice([True, True, True, False])
            )
        return _main_light.node

    # floor/wall randomizer 제거 - USD API (_change_floor_wall_textures)로만 제어
    rep.randomizer.register(randomize_lights, override=True)

    with rep.trigger.on_frame():
        rep.randomizer.randomize_lights()


# ============================================================
# USD API로 팔레트 재질 색상 직접 변경
# ============================================================
_cached_pallet_shaders = None

def _apply_color_to_all_materials(stage, color_rgb):
    global _cached_pallet_shaders
    color = Gf.Vec3f(*color_rgb)

    if _cached_pallet_shaders is None:
        _cached_pallet_shaders = []
        _pallet_texture_inputs = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):
                continue
            path_str = str(prim.GetPath())
            if "Ref_Xform" not in path_str:
                continue
            shader = UsdShade.Shader(prim)
            # diffuse_texture 연결을 한 번만 끊기 (텍스처가 색상을 override하는 문제 방지)
            for tex_name in ("diffuse_texture", "normalmap_texture", "ao_texture"):
                tex_inp = shader.GetInput(tex_name)
                if tex_inp and tex_inp.HasConnectedSource():
                    tex_inp.DisconnectSource()
                    tex_inp.ClearValue()
            # opacity를 1.0으로 강제 (투명도 간섭 제거)
            opacity_inp = shader.GetInput("opacity_constant")
            if opacity_inp:
                opacity_inp.Set(1.0)
            # metallic을 낮게 (금속성이 높으면 diffuse 안 보임)
            metal_inp = shader.GetInput("metallic_constant")
            if metal_inp:
                metal_inp.Set(0.0)
            # 색상 입력 캐시
            tint_inp = shader.GetInput("diffuse_tint")
            if tint_inp:
                _cached_pallet_shaders.append(("tint", tint_inp))
                continue
            for name in ("diffuseColor", "diffuse_color_constant", "base_color_factor"):
                inp = shader.GetInput(name)
                if inp:
                    _cached_pallet_shaders.append(("color", inp))
                    break

    for kind, inp in _cached_pallet_shaders:
        inp.DisconnectSource()
        inp.Set(color)


# ============================================================
# USD API로 바닥/벽 텍스처 직접 변경 (매 프레임)
# ============================================================
_cached_floor_shaders = None
_cached_wall_shaders = None

def _to_omni_uri(filepath):
    """Windows/Linux 파일 경로를 Omniverse가 resolve할 수 있는 file:/// URI로 변환."""
    if filepath.startswith("file:///"):
        return filepath
    filepath = filepath.replace("\\", "/")
    if len(filepath) >= 2 and filepath[1] == ":":
        return f"file:///{filepath}"
    if filepath.startswith("/"):
        return f"file://{filepath}"
    return filepath

def _change_floor_wall_textures(stage, floor_tex_path, wall_tex_path):
    """USD API로 바닥/벽 셰이더의 diffuse_texture를 직접 변경.

    rep.create.material_omnipbr()로 생성된 머티리얼은 /Replicator/Looks/ 하위에 위치.
    Plane Mesh에서 MaterialBindingAPI로 바인딩된 머티리얼의 셰이더를 추적한다.
    """
    global _cached_floor_shaders, _cached_wall_shaders

    if _cached_floor_shaders is None:
        _cached_floor_shaders = []
        _cached_wall_shaders = []

        # Plane Xform들을 찾고, 바인딩된 머티리얼에서 셰이더를 캐시
        plane_idx = 0
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if "/Replicator/Plane" not in path_str:
                continue
            if prim.GetTypeName() != "Xform":
                continue
            parent_path = str(prim.GetParent().GetPath())
            if not (parent_path == "/Replicator" or parent_path.endswith("/Replicator")):
                continue

            is_floor = (plane_idx == 0)
            label = "floor" if is_floor else "wall"
            plane_idx += 1

            # Xform 하위의 Mesh에서 바인딩된 머티리얼 찾기
            for desc in Usd.PrimRange(prim):
                if desc.GetTypeName() != "Mesh":
                    continue
                binding_api = UsdShade.MaterialBindingAPI(desc)
                bound_mat, _ = binding_api.ComputeBoundMaterial()
                if not bound_mat:
                    continue
                mat_prim = bound_mat.GetPrim()
                # 머티리얼 하위에서 셰이더의 diffuse_texture 입력 찾기
                for child in Usd.PrimRange(mat_prim):
                    if not child.IsA(UsdShade.Shader):
                        continue
                    shader = UsdShade.Shader(child)
                    tex_inp = shader.GetInput("diffuse_texture")
                    if not tex_inp:
                        # diffuse_color_constant가 있으면 diffuse_texture 입력 생성
                        if shader.GetInput("diffuse_color_constant"):
                            tex_inp = shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
                    if tex_inp:
                        print(f"    [USD-TEX] {label}: {path_str} -> shader={str(child.GetPath())}")
                        if is_floor:
                            _cached_floor_shaders.append(tex_inp)
                        else:
                            _cached_wall_shaders.append(tex_inp)
                        break
                break  # 첫 번째 Mesh만

        print(f"  [USD-TEX] Cached {len(_cached_floor_shaders)} floor + {len(_cached_wall_shaders)} wall shader inputs")

    # floor 텍스처 설정
    # Omniverse MDL은 Windows 절대 경로를 직접 resolve 가능 (forward slash 사용)
    floor_path_clean = floor_tex_path.replace("\\", "/")
    floor_asset = Sdf.AssetPath(floor_path_clean)
    for tex_inp in _cached_floor_shaders:
        tex_inp.Set(floor_asset)

    # wall 텍스처 설정
    wall_path_clean = wall_tex_path.replace("\\", "/")
    wall_asset = Sdf.AssetPath(wall_path_clean)
    for tex_inp in _cached_wall_shaders:
        tex_inp.Set(wall_asset)
