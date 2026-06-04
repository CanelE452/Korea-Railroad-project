# 03. FSM 연동 — DOPE/YOLO backend + depth fusion + 안전

대상 코드: `25y_automatic_lifter-master/25y_automatic_lifter-master/depth_cam/`

## 전체 경로 (backend 무관 동일)
```
RealSense color+depth → infer_pose(bgr, K, depth_frame) → dict(R, t_m, confirmed, gate_passed, ...)
  → [depth_scale_correct: t를 depth로 보정]
  → [apply_cam_to_fork: 카메라→포크 frame 변환]
  → pose6d_to_align_vars(R, t) → (ψ_pallet, d_lateral, d_forward)
  → FSM.step → CAN → 리프터
```
DOPE/YOLO 둘 다 **동일 dict 계약**이라 위 경로는 backend 무관. (main_rec.py 의 `dope_pose` 변수에 어느 backend든 할당)

## backend 선택
```
config.py POSE_BACKEND = "dope"(기본) | "yolo"   (env override 가능)
  POSE_BACKEND=yolo python main_rec.py
YOLO weights: MODEL_PATH_6D_YOLO (기본 pallet_jetson_deploy/models/pallet_pose_cropaug_v2.pt)
DOPE weights: MODEL_PATH_6D    (challenge/model/challengenight.pth)
```
- 신규 `calib/yolo_inference.py` `YoloPoseEstimator` — DopePoseEstimator와 동일 인터페이스.
  YOLO + 100px pad+shift + SQPnP + sanity gate + N연속 confirm.

## ★ 함정 1 — YOLO→DOPE convention R_fix (필수)
```
DOPE Cuboid3d:               front(0~3) = +Z
YOLO make_pallet_keypoints_3d: near/front(0~3) = −Z (far=+Z)
→ +Z 부호 반대 (Y축 180° 차이). pose6d_adapter는 DOPE(front=+Z) 기준.
해결: R_dope = R_yolo @ diag(-1,1,-1)  (yolo_inference.py 내부 _R_FIX, 자동 적용)
미적용 시 정면 ψ≈±179°(리프터 엉뚱하게 회전), 적용 시 ψ≈0.  t는 centroid라 불변.
검증: forklift frame0(정면) ψ=+3.2°, frame90(회전) ψ=-15.2° — 물리적으로 타당.
```

## ★ 함정 2 — 거리/횡은 monocular라 부정확 → depth fusion (필수)
```
원래: d_forward/d_lateral = monocular PnP의 t (scale 모호). RealSense depth는 gate로만.
실측 오차: monocular PnP z vs RealSense depth z 불일치 평균 ~0.5m(중앙 0.49m, cp07/08/09 79프레임).
          → 포크 삽입엔 치명적 (0.5m면 못 끼움).
해결: pose6d_adapter.depth_scale_correct(t, centroid_uv, depth_m)
      s = depth_m / t.z,  t *= s  (z=측정 depth, x,y 같은 ray 비율로 스케일). R 불변.
      main_rec 의 pose6d_to_align_vars 직전 단일 주입점 → DOPE·YOLO 동시 적용.
검증: 보정 후 t.z == depth (오차 1e-15). depth 무효(0/범위밖)면 무보정 + flag.
유효범위: config DEPTH_CORRECT_Z_MIN_M=0.10 / MAX_M=12.0 (배경 hole 오샘플 방지).
```

## 카메라→포크 extrinsic (골격, 값 실측 필요)
```
config.py CAM_TO_FORK_T = [x,y,z] m,  CAM_TO_FORK_RPY_DEG = [r,p,y] deg  (현재 0 = 항등)
apply_cam_to_fork(R, t): R_fork = Rcf @ R,  t_fork = Rcf @ t + tcf.
카메라가 포크 중심에 없으므로 실측 후 값 채워야 d_lateral/d_forward가 포크 기준이 됨.
```

## 안전 — GATE_PROFILE
```
config.py GATE_PROFILE = "demo"(기본) | "real"   (env override)
              min_kp  reproj  z범위    depth_rel  edge  confirm
  demo        4       50px    0.1~8m   1.00(off)  1.50  1     ← 시연(게이트 거의 off)
  real        6       12px    0.3~6m   0.25       0.40  3     ← 실삽입(나쁜 프레임 거부)
실삽입 전 반드시 GATE_PROFILE=real. snapshot FSM은 한 번 잘못 캡처하면 즉시 잘못된
회전 cmd가 가므로 게이트가 안전판. (STOP_SEC=1.2 인터록도 보조)
```

## 변경 파일 요약
```
신규  calib/yolo_inference.py            YoloPoseEstimator (+R_fix)
변경  calib/pose6d_adapter.py            +depth_scale_correct, +apply_cam_to_fork (기존 함수 유지)
변경  calib/config.py                    +POSE_BACKEND, MODEL_PATH_6D_YOLO, CAM_TO_FORK_*, GATE_PROFILE, DEPTH_CORRECT_*
변경  main_rec.py                        backend 분기 + depth/extrinsic 단일 주입점
미변경 dope_inference.py, fsm/, pose6d_to_align_vars (시그니처 유지)
검증  verify_depth_fusion.py             depth 보정 정량 검증 스크립트
```
세부: `_docs/history/2026-06-04.md`, 메모리 `fsm-yolo-backend`.
