"""sdg_scene.py — Material / Texture 적용 (USD API 직접 조작).

_apply_color_to_all_materials : 팔레트 (Ref_Xform) 셰이더의 diffuse_tint/color 변경
                                + diffuse/normal/ao texture disconnect (텍스처 간섭 방지)
_change_floor_wall_textures   : 바닥/벽 셰이더의 diffuse_texture 를 직접 변경
_to_omni_uri                  : 파일 경로 → Omniverse file:/// URI 변환
"""
import os

from pxr import Gf, Sdf, Usd, UsdShade


# ── 팔레트 색상 적용 ─────────────────────────────────────────────────────
_cached_pallet_shaders = None


def _apply_color_to_all_materials(stage, color_rgb):
    """팔레트 셰이더 (Ref_Xform 포함) 의 diffuse_tint/color 를 일괄 변경.

    diffuse/normal/ao texture 연결 끊고 opacity=1.0, metallic=0.0 강제.
    첫 호출 시 셰이더 캐시 → 매 프레임 빠르게."""
    global _cached_pallet_shaders
    color = Gf.Vec3f(*color_rgb)

    if _cached_pallet_shaders is None:
        _cached_pallet_shaders = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):
                continue
            path_str = str(prim.GetPath())
            if "Ref_Xform" not in path_str:
                continue
            shader = UsdShade.Shader(prim)
            # texture 연결 끊기 (텍스처가 색상을 override 하는 문제 방지)
            for tex_name in ("diffuse_texture", "normalmap_texture", "ao_texture"):
                tex_inp = shader.GetInput(tex_name)
                if tex_inp and tex_inp.HasConnectedSource():
                    tex_inp.DisconnectSource()
                    tex_inp.ClearValue()
            # opacity = 1.0 강제 (투명도 간섭 제거)
            opacity_inp = shader.GetInput("opacity_constant")
            if opacity_inp:
                opacity_inp.Set(1.0)
            metal_inp = shader.GetInput("metallic_constant")
            if metal_inp:
                metal_inp.Set(0.0)
            # 색상 입력 캐시 (tint 우선, 없으면 diffuseColor / diffuse_color_constant / base_color_factor)
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


# ── 바닥/벽 텍스처 변경 ──────────────────────────────────────────────────
_cached_floor_shaders = None
_cached_wall_shaders = None


def _to_omni_uri(filepath):
    """Windows/Linux 파일 경로 → Omniverse 가 resolve 할 수 있는 file:/// URI."""
    if filepath.startswith("file:///"):
        return filepath
    filepath = filepath.replace("\\", "/")
    if len(filepath) >= 2 and filepath[1] == ":":
        return f"file:///{filepath}"
    if filepath.startswith("/"):
        return f"file://{filepath}"
    return filepath


def _change_floor_wall_textures(stage, floor_tex_path, wall_tex_path):
    """USD API 로 바닥/벽 셰이더의 diffuse_texture 직접 변경.

    rep.create.material_omnipbr() 머티리얼은 /Replicator/Looks/ 하위.
    Plane Mesh 의 MaterialBindingAPI 로 바운드 머티리얼 셰이더 추적.
    첫 호출 시 셰이더 캐시 → 매 프레임 빠르게.
    """
    global _cached_floor_shaders, _cached_wall_shaders

    if _cached_floor_shaders is None:
        _cached_floor_shaders = []
        _cached_wall_shaders = []
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

            for desc in Usd.PrimRange(prim):
                if desc.GetTypeName() != "Mesh":
                    continue
                binding_api = UsdShade.MaterialBindingAPI(desc)
                bound_mat, _ = binding_api.ComputeBoundMaterial()
                if not bound_mat:
                    continue
                mat_prim = bound_mat.GetPrim()
                for child in Usd.PrimRange(mat_prim):
                    if not child.IsA(UsdShade.Shader):
                        continue
                    shader = UsdShade.Shader(child)
                    tex_inp = shader.GetInput("diffuse_texture")
                    if not tex_inp:
                        if shader.GetInput("diffuse_color_constant"):
                            tex_inp = shader.CreateInput(
                                "diffuse_texture", Sdf.ValueTypeNames.Asset)
                    if tex_inp:
                        print(f"    [USD-TEX] {label}: {path_str} -> shader={str(child.GetPath())}")
                        (_cached_floor_shaders if is_floor
                         else _cached_wall_shaders).append(tex_inp)
                        break
                break  # 첫 Mesh 만

        print(f"  [USD-TEX] Cached {len(_cached_floor_shaders)} floor + "
              f"{len(_cached_wall_shaders)} wall shader inputs")

    # forward slash 로 (MDL Windows 절대경로 OK)
    floor_asset = Sdf.AssetPath(floor_tex_path.replace("\\", "/"))
    for tex_inp in _cached_floor_shaders:
        tex_inp.Set(floor_asset)

    wall_asset = Sdf.AssetPath(wall_tex_path.replace("\\", "/"))
    for tex_inp in _cached_wall_shaders:
        tex_inp.Set(wall_asset)
