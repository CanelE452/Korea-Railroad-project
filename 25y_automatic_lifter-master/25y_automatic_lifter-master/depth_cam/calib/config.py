# calib/config.py
# 공통 설정 / 상수 / 색상 팔레트 (FSM 다이어그램 및 main_rec.py 실행 흐름 기준)

# ===== 모델 & 감지 =====
MODEL_PATH = 'runs/segment/y11n_seg_finetune/weights/last.pt'  # 실제 가중치 경로에 맞춰 확인 필요
FRONT_CLASS_NAME = "front"
CONF_THR = 0.30  # 세그멘테이션/디텍션 confidence threshold

# ===== 6D pose (DOPE) 모드 =====
# True 면 main_rec.py 가 RealSense color 프레임을 DOPE 추론기에 넣어
# 9 keypoint → PnP → (ψ_pallet, d_lateral, d_forward) 를 계산하고
# fsm.step 의 6D 입력 kwargs (psi_pallet_deg, d_lateral_m, d_forward_m) 로 전달한다.
# False 면 기존 RGB-D RANSAC perception 만 사용.
USE_6D_MODE = True
MODEL_PATH_6D = 'C:/Users/minjae/Documents/github/FoundationPose/challenge/model/challengenight.pth'
# DOPE 입력 height (run_live.py 와 동일하게 400). VGG stride 호환 위해 width 는 자동 정렬.
DOPE_INPUT_HEIGHT = 400
# belief map peak confidence 임계 — challenge/config/task.yaml 의 belief.threshold 와 동일.
DOPE_PEAK_THRESHOLD = 0.30
DOPE_PEAK_SIGMA = 3.0

# ===== 연산 디바이스 / 정밀도 =====
# CUDA가 사용 가능하고 USE_GPU=True면 'cuda:{CUDA_DEVICE}'로 동작.
# USE_HALF=True이면 FP16 추론을 시도합니다(지원되는 GPU에서 속도/메모리 이점).
USE_GPU = True
CUDA_DEVICE = 0
USE_HALF = True

# ===== 깊이 샘플링 / 강건화 =====
SAMPLE_STRIDE = 3          # 마스크 내 픽셀 stride 샘플링 간격
Z_INLIER_THRESH = 0.06     # z(inlier) 허용 오차(m)
MIN_POINTS = 120           # RANSAC에 투입할 최소 3D 포인트 개수

# ===== 평면 적합 (RANSAC) =====
PLANE_INLIER_THRESH = 0.01 # 평면 거리 허용(m)
PLANE_MAX_TRIALS = 200
# ===== EMA 스무딩 =====
EMA_ALPHA_OFFSET = 0.4  # 0.0(없음) ~ 1.0(즉시)
EMA_ALPHA_YAW    = 0.4
EMA_ALPHA_WIDTH  = 0.4

# ===== HUD 색상 (BGR for OpenCV) =====
COLOR_STATUS_OK   = (0, 220, 0)
COLOR_STATUS_TRK  = (0, 165, 255)
COLOR_ALERT       = (0, 0, 255)
COLOR_META        = (180, 180, 0)
COLOR_YAW         = (200, 100, 255)
COLOR_OFFSET      = (50, 200, 255)
COLOR_WIDTH       = (255, 170, 0)
COLOR_ZERR        = (0, 220, 255)
COLOR_BOX         = (0, 180, 255)
COLOR_CNT         = (255, 0, 255)
COLOR_CENTER      = (180, 180, 180)
COLOR_PANEL_BG    = (30, 30, 30)
COLOR_PANEL_EDGE  = (70, 70, 70)

# ===== 정렬 허용치 =====
YAW_TOL_DEG   = 2.00   # ±3 deg
OFF_TOL_M     = 0.12   # ±0.15 m

# ===== 폭(가로 길이) 기준 =====
# detected_length >= WIDTH_MIN_FULL -> ALIGN
# detected_length <  WIDTH_MIN_FULL -> RECOVER
WIDTH_MIN_FULL = 0.00  # m (탐지된 파렛트 최소 길이 임계값)

# ===== ALIGN 단계: 거리 밴드 제어 =====
# 6D pose (DOPE) 모드: ALIGN_DIST_M = 카메라에서 팔레트 centroid (무게중심) 까지 목표 거리 (m).
#   예: 전면 face 2.20m 앞에 정렬하려면 ALIGN_DIST_M = 2.20 + PALLET_DEPTH/2 ≈ 2.85.
# RGB-D + RANSAC 모드 (legacy): 전면 plane 기준이므로 ALIGN_DIST_M = 2.20 (전면 face).
# 두 모드 혼용 시 dist_z 와 d_forward_m 의 기준점이 다름에 유의.
ALIGN_DIST_M = 2.20    # 정렬 목표 거리 (m, 현재 값은 전면 기준)
ALIGN_BAND_M = 0.30    # 허용 밴드 (±m)

# 팔레트 실측 (scan_cleanup/pallet_full.obj). 어댑터 기본값과 동기.
PALLET_WIDTH_M  = 1.10
PALLET_DEPTH_M  = 1.30
PALLET_HEIGHT_M = 0.12

# ===== 포크 hardware 측정 (forklift 자체 spec) =====
# fork center → fork tip 까지 거리 (m). centerToP + forkLen.
# INSERT 단계에서 fork tip 이 pallet 안쪽 어디까지 들어갈지 계산하는 데 사용.
FORK_CENTER_TO_TIP_M = 1.77   # 0.72 (center→P) + 1.05 (P→tip) = 1.77

# ===== INSERT 안전 margin =====
# fork tip 이 pallet back face 에서 띄울 최소 거리 (m).
# 종료 시 fork tip 위치 = entry face + (PALLET_DEPTH_M - INSERT_SAFETY_BACK_M)
# = back face 보다 INSERT_SAFETY_BACK_M 만큼 앞.
INSERT_SAFETY_BACK_M = 0.10

# ===== 정렬 완료 판정 안정화 =====
CMD_STABLE_THR = 5     # 같은 판정이 연속 n프레임 유지돼야 상태 전이

# ===== 회전 목표(각도 기반) =====
# Snapshot 기반 새 다이어그램:
#   - YAW_CORRECT_* / LATERAL_ROTATE_* (첫 회전): snapshot 시점 |ψ_pallet| 만큼 회전.
#     실제 IMU 누적 종료 기준은 max(YAW_TURN_MIN_DEG, snapshot_psi_abs) 또는 그대로.
#   - LATERAL_ROTATE_*_BACK (복귀): LATERAL_BACK_YAW_DEG 만큼 회전.
# 하위 호환: 기존 코드 (구 align.py) 가 REL_YAW_TARGET_DEG 를 참조할 수 있으므로 유지.
REL_YAW_TARGET_DEG    = 85.0   # (legacy) 90° 체인 도달 조건
LATERAL_BACK_YAW_DEG  = 85.0   # *_BACK 복귀 회전 목표 (실측 IMU 노이즈/under-rotate 마진 고려)
YAW_TURN_MIN_DEG      = 1.0    # 첫 회전 최소 보장 (snapshot |ψ| 가 너무 작으면 무시될 수 있어 floor)

# ===== 전역 인터록(정지) =====
# 모든 제어 명령 상태 전환 사이에 STOP을 이 시간만큼 유지합니다.
STOP_SEC = 1.2

# (하위호환) 기존 코드에서 STOP_PAUSE_SEC을 참조하면 STOP_SEC을 사용하도록
STOP_PAUSE_SEC = STOP_SEC

# ===== 전진시간 피팅(가속→정속) 파라미터 =====
# 로그(t_monotonic & dist_z) 기반으로 적합된 파라미터.
# d_acc = 0.5 * FWD_A * FWD_T1^2,  vmax = FWD_A * FWD_T1
# t(d) =
#   d <= d_acc:  FWD_T0 + sqrt(2d/FWD_A)
#   d >  d_acc:  FWD_T0 + FWD_T1 + (d - d_acc)/vmax
USE_PIECEWISE_FWD_FIT = True

FWD_T0   = -0.0202    # s (명령→실이동 시작 지연; ≈0으로 봐도 무방)
FWD_T1   = 4.2780     # s (가속 구간 지속시간)
FWD_A    = 0.071565   # m/s^2 (초기 가속도)
FWD_SCALE = 1.0       # d_eff = FWD_SCALE*|offset| + FWD_BIAS
FWD_BIAS  = 0.0

# 안전 클램프(전진 명령 타이머)
FWD_MIN_SEC = 1.0
FWD_MAX_SEC = 15.0

