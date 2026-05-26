"""sdg_scene.py — 모델 정보 + glTF→USD 변환.

compute_model_info  : USD bbox 측정 + ORIENTATION_OVERRIDES 기반 canonical rotation
convert_gltf_to_usd : 단일 glTF → USD 비동기 변환
convert_all_gltf    : 여러 glTF 일괄 변환 + USD cache + texture verify
_verify_textures    : USD shader 의 texture 파일 누락 시 gltf_dir 에서 자동 copy
"""
import asyncio
import os
import shutil

import numpy as np
import omni.kit.asset_converter
import omni.usd
from pxr import Usd, UsdGeom, UsdShade

from sdg_config import PALLET_TARGET_SIZE
from sdg_math import euler_to_rotation_matrix


# 진단 렌더링으로 확인된 모델별 올바른 회전 (메모리 참조)
# Canonical: X=medium, Y=long, Z=height(up)
ORIENTATION_OVERRIDES = {
    "scene.usd":   (180, 0, 90),    # Z-thin
    "scene_1.usd": (90,  0, 0),     # Y-thin
    "scene_2.usd": (90,  0, 0),
    "scene_3.usd": (90,  0, 90),
}


def compute_model_info(usd_path: str, target_size: float = PALLET_TARGET_SIZE):
    """USD 모델의 canonical bbox + rotation + scale + z_offset 계산.

    Returns: (scale, base_rot, bbox_min, bbox_max, z_offset,
              R_canonical, canonical_bbox_min, canonical_bbox_max)
    """
    stage = Usd.Stage.Open(usd_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    rng = bbox.ComputeAlignedRange()
    mn = rng.GetMin(); mx = rng.GetMax()
    size = mx - mn
    dims = [size[0], size[1], size[2]]
    longest = max(dims)
    scale = target_size / longest if longest > 0 else 1.0

    bbox_min = [float(mn[0]), float(mn[1]), float(mn[2])]
    bbox_max = [float(mx[0]), float(mx[1]), float(mx[2])]

    min_idx = dims.index(min(dims))
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

    # 메시 노멀 분석 (fallback 용 — 팔레트는 ORIENTATION_OVERRIDES 우선)
    thin_axis = min_idx
    plus_count = minus_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        normals = mesh.GetNormalsAttr().Get()
        if normals is None:
            continue
        for n in normals:
            val = n[thin_axis]
            if abs(val) > 0.5:
                if val > 0:
                    plus_count += 1
                else:
                    minus_count += 1
    print(f"  {os.path.basename(usd_path)}: normal analysis axis={'XYZ'[thin_axis]}: "
          f"+{plus_count} / -{minus_count}")

    basename = os.path.basename(usd_path)
    if basename in ORIENTATION_OVERRIDES:
        base_rot = ORIENTATION_OVERRIDES[basename]
        print(f"  {basename}: using OVERRIDE rotation {base_rot}")
    else:
        # fallback: 노멀 분석 기반
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
    # Y↔Z swap: Rx(-90°)
    R_yz_swap = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    R_canonical = R_yz_swap @ R_canonical
    rotated = (R_canonical @ corners.T).T

    min_y = rotated[:, 1].min()
    z_offset = -min_y * scale

    canon_min = rotated.min(axis=0)
    canon_max = rotated.max(axis=0)
    canonical_bbox_min = [float(canon_min[0]), float(canon_min[1]), float(canon_min[2])]
    canonical_bbox_max = [float(canon_max[0]), float(canon_max[1]), float(canon_max[2])]

    print(f"  {os.path.basename(usd_path)}: dims=({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f}) "
          f"thin={'XYZ'[min_idx]} base_rot={base_rot} scale={scale:.6f} z_offset={z_offset:.4f}")
    print(f"    canonical bbox: min={canonical_bbox_min} max={canonical_bbox_max}")
    return (scale, base_rot, bbox_min, bbox_max, z_offset,
            R_canonical, canonical_bbox_min, canonical_bbox_max)


# ── glTF → USD 변환 ──────────────────────────────────────────────────────

async def convert_gltf_to_usd(input_path: str, output_path: str) -> bool:
    """단일 glTF → USD 비동기 변환."""
    ctx = omni.kit.asset_converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_cameras = True
    ctx.single_mesh = False
    ctx.smooth_normals = True
    ctx.use_meter_as_world_unit = True

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        input_path, output_path, progress_callback=None, asset_converter_context=ctx)
    success = await task.wait_until_finished()
    if not success:
        print(f"[ERROR] convert fail {input_path}: "
              f"{task.get_status()} - {task.get_detailed_error()}")
    return success


def _verify_textures(usd_path: str, gltf_dir: str):
    """USD shader 의 texture 파일 누락 시 gltf_dir 에서 자동 copy."""
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
                    abs_tex = (tex_ref if os.path.isabs(tex_ref)
                               else os.path.normpath(os.path.join(usd_dir, tex_ref)))
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
    """여러 glTF 일괄 변환. USD 캐시 있으면 skip, 없으면 변환 + texture verify."""
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
