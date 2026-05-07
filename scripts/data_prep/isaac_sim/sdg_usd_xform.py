"""USD xformOp 기반 pose/light/visibility 제어 함수.

gen_replicator_data.py에서 분리된 USD prim 직접 조작 유틸리티.
Replicator 그래프 노드 누적을 방지하기 위해 USD API를 직접 사용한다.
"""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from sdg_math import euler_to_rotation_matrix, rotation_matrix_to_quat_xyzw


_distractor_prim_path_cache = {}  # distractor index -> USD prim path
_rep_prim_path_cache = {}  # id(rep_prim) -> USD prim path (generic)

def _resolve_rep_prim_path(rep_prim):
    """Replicator 객체에서 USD prim path를 추출 (generic, 캐싱)."""
    _id = id(rep_prim)
    if _id in _rep_prim_path_cache:
        return _rep_prim_path_cache[_id]
    _path = None
    if hasattr(rep_prim, 'node'):
        try:
            _path = rep_prim.node.get_prim_path()
        except Exception:
            pass
    if not _path and hasattr(rep_prim, 'get_output_prims'):
        try:
            _out = rep_prim.get_output_prims()
            if _out and len(_out) > 0:
                _path = str(_out[0].GetPath()) if hasattr(_out[0], 'GetPath') else str(_out[0])
        except Exception:
            pass
    _rep_prim_path_cache[_id] = _path
    return _path

def _resolve_distractor_prim_path(distractor_prim):
    """Replicator 객체에서 USD prim path를 추출."""
    _id = id(distractor_prim)
    if _id in _distractor_prim_path_cache:
        return _distractor_prim_path_cache[_id]
    _path = _resolve_rep_prim_path(distractor_prim)
    _distractor_prim_path_cache[_id] = _path
    return _path


# --- USD xformOp 기반 pose 설정 (Replicator 그래프 노드 누적 방지) ---
_xformable_cache = {}  # prim_path -> (UsdGeom.Xformable, translate_op, orient_op, scale_op)

def _set_pose_usd(stage, prim_path, position=None, rotation_deg=None, scale=None):
    """USD xformOp API로 pose를 직접 설정.

    Replicator가 생성한 prim은 xformOp:translate, xformOp:orient, xformOp:scale을 가짐.
    이 함수는 기존 ops의 값만 업데이트하여 그래프 노드 누적을 방지한다.

    Args:
        stage: USD stage
        prim_path: USD prim path (str)
        position: (x, y, z) tuple or None
        rotation_deg: (rx, ry, rz) XYZ intrinsic euler degrees or None
        scale: (sx, sy, sz) tuple or single float or None
    """
    if prim_path is None:
        return False

    if prim_path in _xformable_cache:
        xformable, t_op, o_op, s_op = _xformable_cache[prim_path]
    else:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        xformable = UsdGeom.Xformable(prim)

        # Replicator는 보통 translate, orient(quaternion), scale 순서로 xformOps를 생성
        t_op = o_op = s_op = None
        for op in xformable.GetOrderedXformOps():
            op_name = op.GetName()
            if "translate" in op_name and t_op is None:
                t_op = op
            elif "orient" in op_name and o_op is None:
                o_op = op
            elif "scale" in op_name and s_op is None:
                s_op = op

        # ops가 없으면 새로 생성 (처음 1회만)
        if t_op is None:
            t_op = xformable.AddTranslateOp()
        if o_op is None:
            o_op = xformable.AddOrientOp()
        if s_op is None:
            s_op = xformable.AddScaleOp()

        _xformable_cache[prim_path] = (xformable, t_op, o_op, s_op)

    if position is not None:
        t_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if rotation_deg is not None:
        # XYZ intrinsic euler → quaternion (Replicator convention과 동일)
        R = euler_to_rotation_matrix(rotation_deg)
        qx, qy, qz, qw = rotation_matrix_to_quat_xyzw(R)
        o_op.Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))

    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = (scale, scale, scale)
        s_op.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

    return True


def _set_pose_usd_rep(stage, rep_prim, position=None, rotation_deg=None, scale=None):
    """Replicator prim 객체를 받아 USD xformOp로 pose 설정 (prim path 자동 resolve)."""
    prim_path = _resolve_rep_prim_path(rep_prim)
    return _set_pose_usd(stage, prim_path, position, rotation_deg, scale)


def _set_camera_look_at_usd(stage, camera_rep_prim, position, look_at_target, up=(0, 0, 1)):
    """USD xformOp API로 카메라 pose를 look_at 방식으로 직접 설정.

    Isaac Sim 카메라 convention: -Z forward, +Y up (OpenGL style).
    World convention: +Z up.

    Args:
        stage: USD stage
        camera_rep_prim: Replicator 카메라 객체
        position: (x, y, z) 카메라 월드 위치
        look_at_target: (x, y, z) look-at 대상 월드 위치
        up: world up 벡터 (default: Z-up)
    """
    cam_pos = np.array(position, dtype=np.float64)
    target = np.array(look_at_target, dtype=np.float64)
    up_vec = np.array(up, dtype=np.float64)

    # Camera-to-world 행렬 계산
    # forward = target - cam_pos (world space에서 카메라가 바라보는 방향)
    forward = target - cam_pos
    fwd_len = np.linalg.norm(forward)
    if fwd_len < 1e-10:
        return False
    forward = forward / fwd_len

    right = np.cross(forward, up_vec)
    right_len = np.linalg.norm(right)
    if right_len < 1e-10:
        # forward가 up과 평행한 경우 fallback
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_len
    cam_up = np.cross(right, forward)

    # Isaac Sim 카메라: -Z = forward, +Y = up, +X = right
    # Camera-to-world rotation: columns = [right, cam_up, -forward]
    R_c2w = np.column_stack([right, cam_up, -forward])

    # Rotation matrix -> quaternion (wxyz for USD)
    qx, qy, qz, qw = rotation_matrix_to_quat_xyzw(R_c2w)

    # USD prim path resolve
    prim_path = _resolve_rep_prim_path(camera_rep_prim)
    if prim_path is None:
        return False

    if prim_path in _xformable_cache:
        xformable, t_op, o_op, s_op = _xformable_cache[prim_path]
    else:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        xformable = UsdGeom.Xformable(prim)
        t_op = o_op = s_op = None
        for op in xformable.GetOrderedXformOps():
            op_name = op.GetName()
            if "translate" in op_name and t_op is None:
                t_op = op
            elif "orient" in op_name and o_op is None:
                o_op = op
            elif "scale" in op_name and s_op is None:
                s_op = op
        if t_op is None:
            t_op = xformable.AddTranslateOp()
        if o_op is None:
            o_op = xformable.AddOrientOp()
        if s_op is None:
            s_op = xformable.AddScaleOp()
        _xformable_cache[prim_path] = (xformable, t_op, o_op, s_op)

    t_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
    o_op.Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    return True


def _randomize_lights_usd(stage, rng, prebuilt):
    """USD API로 조명을 직접 랜덤화 (OmniGraph 경유 없음).

    DomeLight: intensity, color, HDRI
    RectLight x3: position, intensity, color, visibility
    """
    dome_light = prebuilt["dome_light"]

    # DomeLight
    dome_intensity = float(rng.uniform(2000, 3500))
    dome_r = float(rng.uniform(0.70, 1.0))
    dome_g = float(rng.uniform(0.70, 1.0))
    dome_b = float(rng.uniform(0.70, 1.0))
    dome_light.GetIntensityAttr().Set(dome_intensity)
    dome_light.GetColorAttr().Set(Gf.Vec3f(dome_r, dome_g, dome_b))
    hdri_files = prebuilt.get("hdri_files", [])
    if hdri_files:
        hdri_idx = int(rng.integers(len(hdri_files)))
        dome_light.GetTextureFileAttr().Set(hdri_files[hdri_idx])

    # RectLights (main, fill1, fill2)
    light_prims = prebuilt["lights"]

    # Main light
    _set_pose_usd_rep(stage, light_prims[0],
                      position=(float(rng.uniform(-3, 3)),
                                float(rng.uniform(-3, 3)),
                                float(rng.uniform(4, 6))),
                      rotation_deg=(float(rng.uniform(-100, -70)),
                                    float(rng.uniform(-20, 20)),
                                    float(rng.uniform(-20, 20))))
    _set_light_attrs_usd(stage, light_prims[0],
                         intensity=float(rng.uniform(100000, 300000)),
                         color=(float(rng.uniform(0.85, 1.0)),
                                float(rng.uniform(0.82, 0.98)),
                                float(rng.uniform(0.78, 0.95))))

    # Fill light 1
    _set_pose_usd_rep(stage, light_prims[1],
                      position=(float(rng.uniform(-5, 5)),
                                float(rng.uniform(-5, 5)),
                                float(rng.uniform(2, 5))))
    _set_light_attrs_usd(stage, light_prims[1],
                         intensity=float(rng.uniform(50000, 200000)),
                         color=(float(rng.uniform(0.7, 1.0)),
                                float(rng.uniform(0.7, 1.0)),
                                float(rng.uniform(0.75, 0.95))),
                         visible=(float(rng.random()) < 0.75))

    # Fill light 2
    _set_pose_usd_rep(stage, light_prims[2],
                      position=(float(rng.uniform(-5, 5)),
                                float(rng.uniform(-5, 5)),
                                float(rng.uniform(2, 5))))
    _set_light_attrs_usd(stage, light_prims[2],
                         intensity=float(rng.uniform(40000, 180000)),
                         color=(float(rng.uniform(0.75, 1.0)),
                                float(rng.uniform(0.75, 1.0)),
                                float(rng.uniform(0.75, 0.95))),
                         visible=(float(rng.random()) < 0.75))


_light_prim_cache = {}  # id(rep_light) -> UsdLux prim


def _set_light_attrs_usd(stage, rep_light, intensity=None, color=None, visible=None):
    """USD API로 RectLight 속성을 직접 설정."""
    _id = id(rep_light)
    if _id not in _light_prim_cache:
        prim_path = _resolve_rep_prim_path(rep_light)
        if not prim_path:
            return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        # RectLight prim 자체가 아닐 수 있음 — 하위에서 RectLight 검색
        light_prim = None
        if prim.IsA(UsdLux.RectLight):
            light_prim = prim
        else:
            for desc in Usd.PrimRange(prim):
                if desc.IsA(UsdLux.RectLight):
                    light_prim = desc
                    break
        if not light_prim:
            return
        _light_prim_cache[_id] = light_prim

    lp = _light_prim_cache[_id]
    rect = UsdLux.RectLight(lp)

    if intensity is not None:
        rect.GetIntensityAttr().Set(intensity)
    if color is not None:
        rect.GetColorAttr().Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
    if visible is not None:
        imageable = UsdGeom.Imageable(lp)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _set_distractor_visible(stage, distractor_prim, visible):
    """v10: USD API로 visibility 직접 설정 (Replicator 그래프 노드 누적 방지)."""
    try:
        _dp_path = _resolve_distractor_prim_path(distractor_prim)
        if not _dp_path:
            return
        _dp_prim = stage.GetPrimAtPath(_dp_path)
        if _dp_prim and _dp_prim.IsValid():
            from pxr import UsdGeom as _UG_vis
            if visible:
                _UG_vis.Imageable(_dp_prim).MakeVisible()
            else:
                _UG_vis.Imageable(_dp_prim).MakeInvisible()
    except Exception:
        pass


def _apply_distractor_color(stage, distractor_idx, color_rgb, shader_cache, rng=None):
    """v10: USD API로 기존 머티리얼의 파라미터만 변경 (새 prim 생성 없음)."""
    if distractor_idx >= len(shader_cache) or shader_cache[distractor_idx] is None:
        return
    shader = shader_cache[distractor_idx]
    from pxr import Gf
    _inp = shader.GetInput("diffuse_color_constant")
    if _inp:
        _inp.Set(Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])))
    _r_inp = shader.GetInput("reflection_roughness_constant")
    if _r_inp:
        _r_inp.Set(float(rng.uniform(0.3, 0.9)) if rng is not None else float(np.random.uniform(0.3, 0.9)))
    _m_inp = shader.GetInput("metallic_constant")
    if _m_inp:
        _m_inp.Set(float(rng.uniform(0.0, 0.15)) if rng is not None else float(np.random.uniform(0.0, 0.15)))
