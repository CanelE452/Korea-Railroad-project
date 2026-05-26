"""
twin-PnP contract 검증 (dual-debate 합의문 P0-1)

목적
----
DOPE 추론에서 사용할 3D keypoint contract 둘 중 어느 쪽이
synthetic 학습 데이터(NDDS JSON)와 일치하는지 확정한다.

- (A) **default**: `Deep_Object_Pose/common/cuboid.py: Cuboid3d(dim_cm).get_vertices()`
       — OpenCV convention (X=right, Y=down, Z=forward)
       — front face {0,1,2,3} at Z = +depth/2
       — top face는 Y = -height/2 (Y down 기준)

- (B) **isaac**: `scripts/self_training/pnp_solver.py: make_pallet_keypoints_3d_isaac()`
       — Isaac canonical ordering (synthetic data와 일치하도록 작성됨)
       — front face {0,1,2,3} at Z = +depth/2
       — 좌표 부호 다름 (left/right, top/bottom)

같은 2D projected_cuboid + intrinsic으로 두 가지 PnP를 풀고
NDDS JSON의 GT pose_transform 과 비교한다. **GT와 축이 일치하는 쪽이 정답.**

사용법
------
    conda activate pallet-pose
    cd C:/Users/minjae/Documents/github/FoundationPose
    python depth_cam/tools/twin_pnp_check.py \\
        --json data/pallet/training_data/blender_dark/000000.json

여러 프레임으로 검증 (강력 권장):
    python depth_cam/tools/twin_pnp_check.py \\
        --json_glob "data/pallet/training_data/blender_dark/00000*.json" \\
        --n 10

출력
----
프레임별로 default/isaac 각각의:
  - GT 와의 R 축 dot product (R @ e_i 의 일치도, 1.0이면 완벽)
  - translation 오차 (m)
  - reproj error (px)

마지막에 평균 점수 + 권장 결론.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Deep_Object_Pose/common 을 sys.path 에 추가 (Cuboid3d, CuboidPNPSolver)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Deep_Object_Pose", "common"))
from cuboid import Cuboid3d  # noqa: E402

# scripts/self_training 의 isaac 9점 함수
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "self_training"))
from pnp_solver import make_pallet_keypoints_3d_isaac  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────────────────────────

def load_ndds_frame(path: str) -> dict:
    """NDDS JSON 1프레임 로드, 필요한 필드만 추출."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("objects"):
        raise ValueError(f"{path}: no objects in JSON")
    obj = data["objects"][0]

    # camera intrinsic
    cam = data["camera_data"]["intrinsics"]
    K = np.array([
        [cam["fx"], 0.0,       cam["cx"]],
        [0.0,       cam["fy"], cam["cy"]],
        [0.0,       0.0,       1.0],
    ], dtype=np.float64)

    # 2D keypoints: 8 corners + centroid → 9개
    proj8 = np.array(obj["projected_cuboid"], dtype=np.float64)          # (8, 2)
    proj_c = np.array(obj["projected_cuboid_centroid"], dtype=np.float64)  # (2,)
    kpts_2d = np.vstack([proj8, proj_c[None, :]])                         # (9, 2)

    # GT pose_transform: 4x4, object-frame → camera-frame, 단위 m
    pose = np.array(obj["pose_transform"], dtype=np.float64)              # (4, 4)
    R_gt = pose[:3, :3]
    t_gt_m = pose[:3, 3]

    # location (NDDS는 m 단위, GT 와 동일)
    loc_m = np.array(obj["location"], dtype=np.float64)

    return {
        "K": K,
        "kpts_2d": kpts_2d,
        "R_gt": R_gt,
        "t_gt_m": t_gt_m,
        "loc_m": loc_m,
        "json_path": path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PnP 솔버 (두 가지 contract)
# ──────────────────────────────────────────────────────────────────────────────

def solve_pnp_with_contract(
    kpts_2d: np.ndarray,        # (9, 2)
    kpts_3d: np.ndarray,        # (9, 3), 단위 일관
    K: np.ndarray,              # (3, 3)
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """EPnP로 (R, t) 추정. t의 단위는 kpts_3d 의 단위와 동일."""
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=kpts_3d.astype(np.float64),
            imagePoints=kpts_2d.astype(np.float64),
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        return R, t
    except cv2.error:
        return None


def get_default_keypoints_3d_cm(width_m: float, depth_m: float, height_m: float) -> np.ndarray:
    """Cuboid3d 기본 contract의 9점 (cm)."""
    cuboid = Cuboid3d([width_m * 100.0, height_m * 100.0, depth_m * 100.0])
    return np.array(cuboid.get_vertices(), dtype=np.float64)  # (9, 3) cm


def get_default_z180_keypoints_3d_cm(width_m: float, depth_m: float, height_m: float) -> np.ndarray:
    """default Cuboid3d 의 X/Y 부호 반전 (= Z축 180° 회전한 contract)."""
    pts = get_default_keypoints_3d_cm(width_m, depth_m, height_m)
    return pts @ np.diag([-1.0, -1.0, 1.0])


def get_default_x_flip_keypoints_3d_cm(width_m: float, depth_m: float, height_m: float) -> np.ndarray:
    """default 의 X 부호만 반전."""
    pts = get_default_keypoints_3d_cm(width_m, depth_m, height_m)
    return pts @ np.diag([-1.0, 1.0, 1.0])


def get_default_y_flip_keypoints_3d_cm(width_m: float, depth_m: float, height_m: float) -> np.ndarray:
    """default 의 Y 부호만 반전."""
    pts = get_default_keypoints_3d_cm(width_m, depth_m, height_m)
    return pts @ np.diag([1.0, -1.0, 1.0])


# ──────────────────────────────────────────────────────────────────────────────
# 비교 메트릭
# ──────────────────────────────────────────────────────────────────────────────

def axis_alignment_scores(R_pred: np.ndarray, R_gt: np.ndarray) -> dict:
    """예측 R 의 각 축이 GT R 의 같은 축과 얼마나 일치하는가.

    dot이 +1이면 완벽 일치, -1이면 정반대, 0이면 90도 어긋남.
    """
    axes = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    return {name: float(np.dot(R_pred @ v, R_gt @ v)) for name, v in axes.items()}


def reproj_error_px(kpts_3d: np.ndarray, kpts_2d: np.ndarray,
                    R: np.ndarray, t: np.ndarray, K: np.ndarray) -> float:
    """PnP solution으로 projection 후 ground truth 2D와의 평균 픽셀 거리."""
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    proj, _ = cv2.projectPoints(
        kpts_3d.astype(np.float64), rvec, tvec, K, np.zeros((4, 1), dtype=np.float64)
    )
    proj = proj.reshape(-1, 2)
    errs = np.linalg.norm(proj - kpts_2d, axis=1)
    return float(np.mean(errs))


# ──────────────────────────────────────────────────────────────────────────────
# 1 프레임 비교
# ──────────────────────────────────────────────────────────────────────────────

def compare_one_frame(frame: dict, width_m: float, depth_m: float, height_m: float,
                      verbose: bool = True) -> dict:
    """1 프레임에서 default vs isaac 비교."""
    K = frame["K"]
    kpts_2d = frame["kpts_2d"]
    R_gt = frame["R_gt"]
    t_gt_m = frame["t_gt_m"]
    name = os.path.basename(frame["json_path"])

    results = {}

    # 비교할 contract 후보들 — 모두 동일 단위 처리. cm 또는 m.
    contracts = {
        "default":       (get_default_keypoints_3d_cm(width_m, depth_m, height_m),       "cm"),
        "default_z180":  (get_default_z180_keypoints_3d_cm(width_m, depth_m, height_m),  "cm"),
        "default_xflip": (get_default_x_flip_keypoints_3d_cm(width_m, depth_m, height_m), "cm"),
        "default_yflip": (get_default_y_flip_keypoints_3d_cm(width_m, depth_m, height_m), "cm"),
        "isaac":         (make_pallet_keypoints_3d_isaac(width=width_m, depth=depth_m, height=height_m), "m"),
    }

    for tag, (kpts_3d, unit) in contracts.items():
        sol = solve_pnp_with_contract(kpts_2d, kpts_3d, K)
        if sol is None:
            results[tag] = None
            continue
        R, t = sol
        t_m = t / 100.0 if unit == "cm" else t
        results[tag] = {
            "R": R, "t_m": t_m,
            "axes": axis_alignment_scores(R, R_gt),
            "t_err_m": float(np.linalg.norm(t_m - t_gt_m)),
            "reproj_px": reproj_error_px(kpts_3d, kpts_2d, R, t, K),
        }

    if verbose:
        print(f"\n--[{name}]--")
        print(f"  GT           t (m): [{t_gt_m[0]:+.3f}, {t_gt_m[1]:+.3f}, {t_gt_m[2]:+.3f}]")
        for tag, r in results.items():
            if r is None:
                print(f"  {tag:14s}: PnP failed")
                continue
            t = r["t_m"]
            axes = r["axes"]
            print(f"  {tag:14s}: t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]  "
                  f"|dt|={r['t_err_m']:.3f}m  reproj={r['reproj_px']:6.2f}px  "
                  f"axes_dot(X/Y/Z)={axes['X']:+.3f}/{axes['Y']:+.3f}/{axes['Z']:+.3f}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 최종 판정
# ──────────────────────────────────────────────────────────────────────────────

def summarize(all_results: List[dict]) -> None:
    """여러 프레임 결과를 평균내서 어느 contract가 GT와 일치하는지 판정."""

    def collect(tag: str):
        rs = [r[tag] for r in all_results if r.get(tag) is not None]
        if not rs:
            return None
        return {
            "n_ok": len(rs),
            "mean_reproj_px": float(np.mean([r["reproj_px"] for r in rs])),
            "mean_t_err_m": float(np.mean([r["t_err_m"] for r in rs])),
            "mean_axis_X": float(np.mean([r["axes"]["X"] for r in rs])),
            "mean_axis_Y": float(np.mean([r["axes"]["Y"] for r in rs])),
            "mean_axis_Z": float(np.mean([r["axes"]["Z"] for r in rs])),
        }

    contract_tags = ["default", "default_z180", "default_xflip", "default_yflip", "isaac"]
    summaries = {tag: collect(tag) for tag in contract_tags}

    print("\n" + "=" * 90)
    print(" 최종 판정 (모든 프레임 평균)")
    print("=" * 90)

    def show(tag: str, s):
        if s is None:
            print(f"  {tag:14s}: PnP 모두 실패")
            return
        print(f"  {tag:14s}: ok {s['n_ok']:3d}/{len(all_results)}  "
              f"reproj={s['mean_reproj_px']:6.2f}px  |dt|={s['mean_t_err_m']:.3f}m  "
              f"axes_dot(X/Y/Z)={s['mean_axis_X']:+.3f}/{s['mean_axis_Y']:+.3f}/{s['mean_axis_Z']:+.3f}")

    for tag in contract_tags:
        show(tag, summaries[tag])

    # 판정 규칙: 모든 축의 dot이 +0.9 이상이면 "GT와 동일 contract"
    THRESHOLD_DOT = 0.9
    THRESHOLD_T = 0.10  # 10cm
    print()

    def verdict(tag: str, s):
        if s is None:
            return f"{tag}: 사용 불가 (PnP 모두 실패)"
        axes_ok = (s['mean_axis_X'] > THRESHOLD_DOT and
                   s['mean_axis_Y'] > THRESHOLD_DOT and
                   s['mean_axis_Z'] > THRESHOLD_DOT)
        t_ok = s['mean_t_err_m'] < THRESHOLD_T
        if axes_ok and t_ok:
            return f"{tag}: [OK] GT와 일치 (축 dot > {THRESHOLD_DOT} & |dt| < {THRESHOLD_T}m)"
        # 축이 일부만 일치 → flipped contract 가능성
        axis_problems = []
        for ax in "XYZ":
            v = s[f"mean_axis_{ax}"]
            if v < -0.5:
                axis_problems.append(f"{ax} 뒤집힘({v:+.2f})")
            elif v < THRESHOLD_DOT:
                axis_problems.append(f"{ax} 어긋남({v:+.2f})")
        why = ", ".join(axis_problems) if axis_problems else f"|dt|={s['mean_t_err_m']:.2f}m"
        return f"{tag}: [FAIL] GT 불일치 ({why})"

    for tag in contract_tags:
        print("  " + verdict(tag, summaries[tag]))

    print()
    print("-" * 90)
    print(" 권장 결론")
    print("-" * 90)

    # 통과한 contract 중 reproj가 가장 작은 것 = 정답
    passed = []
    for tag in contract_tags:
        s = summaries[tag]
        if s is None:
            continue
        axes_ok = (s['mean_axis_X'] > THRESHOLD_DOT and
                   s['mean_axis_Y'] > THRESHOLD_DOT and
                   s['mean_axis_Z'] > THRESHOLD_DOT)
        t_ok = s['mean_t_err_m'] < THRESHOLD_T
        if axes_ok and t_ok:
            passed.append((tag, s))

    if passed:
        passed.sort(key=lambda x: x[1]['mean_reproj_px'])
        winner_tag, winner = passed[0]
        print(f"  → {winner_tag} 사용. (reproj {winner['mean_reproj_px']:.2f}px, |dt|={winner['mean_t_err_m']:.3f}m)")
        if winner_tag == "default":
            print("    depth_cam DOPE perception에서 Cuboid3d(dim_cm) 기본 그대로.")
        elif winner_tag == "isaac":
            print("    `from pnp_solver import make_pallet_keypoints_3d_isaac` 사용.")
        elif winner_tag == "default_z180":
            print("    Cuboid3d(dim_cm).get_vertices() @ diag([-1,-1,+1]) 사용.")
            print("    의미: 학습 데이터의 corner 0이 default의 corner 2 (180° 회전).")
        elif winner_tag == "default_xflip":
            print("    Cuboid3d(dim_cm).get_vertices() @ diag([-1,+1,+1]) 사용.")
            print("    의미: 학습 데이터가 X 부호 반전된 frame.")
        elif winner_tag == "default_yflip":
            print("    Cuboid3d(dim_cm).get_vertices() @ diag([+1,-1,+1]) 사용.")
            print("    의미: 학습 데이터가 Y 부호 반전된 frame (= Y-up convention).")
    else:
        print("  → 통과 contract 없음. 다음 중 하나:")
        print("     (a) JSON projected_cuboid 순서가 가정한 것과 다름")
        print("     (b) 다른 keypoint convention (e.g., R_yz_swap)")
        print("     (c) pose_transform 의 R 정의가 OpenCV가 아닌 다른 frame")
        print("    더 많은 프레임으로 재실행, 또는 cuboid raw vertex 좌표 직접 인쇄.")
    print("=" * 90)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="단일 NDDS JSON 경로")
    ap.add_argument("--json_glob", default=None, help="여러 JSON glob 패턴 (--n 과 함께)")
    ap.add_argument("--n", type=int, default=10, help="--json_glob 사용 시 최대 프레임 수")
    ap.add_argument("--width",  type=float, default=1.1,  help="팔레트 폭 m (default 1.1)")
    ap.add_argument("--depth",  type=float, default=1.3,  help="팔레트 깊이 m (default 1.3)")
    ap.add_argument("--height", type=float, default=0.11, help="팔레트 높이 m (default 0.11)")
    ap.add_argument("--quiet", action="store_true", help="프레임별 상세 출력 생략, 최종 요약만")
    args = ap.parse_args()

    # 입력 수집
    if args.json:
        paths = [args.json]
    elif args.json_glob:
        paths = sorted(glob.glob(args.json_glob))[:args.n]
        if not paths:
            print(f"[ERROR] glob 매칭 0건: {args.json_glob}")
            sys.exit(1)
    else:
        # 기본: blender_dark 첫 5장
        paths = sorted(glob.glob(os.path.join(
            _REPO_ROOT, "data", "pallet", "training_data", "blender_dark", "*.json"
        )))[:5]
        if not paths:
            print("[ERROR] 기본 경로에 NDDS JSON 없음. --json 또는 --json_glob 명시.")
            sys.exit(1)

    print(f"[INFO] {len(paths)} 프레임 분석. pallet dim = (W={args.width}, D={args.depth}, H={args.height}) m")

    all_results = []
    for p in paths:
        try:
            frame = load_ndds_frame(p)
        except Exception as e:
            print(f"[WARN] {p}: 로드 실패 — {e}")
            continue
        r = compare_one_frame(frame, args.width, args.depth, args.height, verbose=not args.quiet)
        all_results.append(r)

    if not all_results:
        print("[ERROR] 유효한 프레임 0건.")
        sys.exit(1)

    summarize(all_results)


if __name__ == "__main__":
    main()
