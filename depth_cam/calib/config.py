# calib/config.py
# 공통 설정 / 상수 / 색상 팔레트 (FSM 다이어그램 및 main_rec.py 실행 흐름 기준)

# ===== 모델 & 감지 =====
# 6D pose 통합 (2026-05-22). 이전 YOLO seg 경로는 _legacy 로 보존.
MODEL_PATH = r'C:/Users/minjae/Documents/github/FoundationPose/challenge/weights/baseline_v8_A.pth'
MODEL_PATH_YOLO_LEGACY = 'runs/segment/y11n_seg_finetune/weights/last.pt'
FRONT_CLASS_NAME = "front"   # YOLO legacy 호환 (DOPEPerception은 사용 안 함)
CONF_THR = 0.30              # YOLO legacy 호환

# ===== DOPE 추론 (challenge/config/task.yaml 과 일치) =====
DOPE_BELIEF_THRESHOLD = 0.30
DOPE_BELIEF_THRESH_MAP = 0.30
DOPE_BELIEF_THRESH_POINTS = 0.30
DOPE_BELIEF_THRESH_ANGLE = 0.50
DOPE_BELIEF_SIGMA = 3
DOPE_INPUT_HEIGHT = 400          # run_live.py 와 동일한 전처리 (h=400)
DOPE_GATE_MIN_KP = 5            # 7 → 5 (capturepallet03 detection 안정성 위해 완화)
DOPE_GATE_MAX_REPROJ_PX = 8.0
DOPE_GATE_Z_MIN_M = 0.30
DOPE_GATE_Z_MAX_M = 7.00            # 5.0 → 7.0 (capturepallet03 멀리 있는 frame 포함)
DOPE_GATE_DEPTH_PNP_REL = 0.50      # 0.30 → 0.50 (outdoor depth noise 완화)
DOPE_TEMPORAL_CONFIRM_FRAMES = 2

# ===== 팔레트 dim (twin_pnp_check 검증값, 2026-05-22) =====
# 학습 데이터 (mixed_v8_train) 의 실제 cuboid dim.
# task.yaml 의 (1.1, 1.3, 0.11) 은 spec 값이고 실제 학습 라벨과 다름.
# 50/50 프레임에서 default_z180 contract + 아래 dim 으로 reproj=2.89px, |dt|=0.085m 확인.
PALLET_WIDTH_M  = 1.0    # 앞면 폭 (X)
PALLET_DEPTH_M  = 1.2    # 깊이 (Z, front 방향)
PALLET_HEIGHT_M = 0.15   # 두께 (Y)

# PnP contract: default Cuboid3d 의 vertices @ diag([-1,-1,+1]).
# 의미: 학습 데이터의 corner 0 이 default 의 corner 2 (Z축 180° 회전).
PALLET_PNP_CONTRACT_Z180 = True

# ===== yaw / depth direction 보정 (manual GT 27 frame 분석으로 확정, 2026-05-22) =====
# `depth_cam/tools/calibrate_from_manual_gt.py` 결과:
#   - 진짜 정렬 frame (R[2,2]<0, 카메라가 정면 봄) = `1778652176547299328.json`
#   - 그 frame 에서 식 A (atan2(front[0], -front[2])) = +4.14°  → 거의 0, 식 A 정답
#   - 그 frame 에서 식 B (atan2(front[0], +front[2])) = +175.86°  → 정렬 아님
# capturepallet07 의 27 frame 평균은 뒷면 frame 들이 섞여 흐려짐. 평균값 사용 금지.
# 시연 직전 보정 거의 불필요 (정렬 시 ±5° 오차, EMA smoothing 흡수).
YAW_OFFSET_DEG = 0.0       # 정렬 시 yaw_smooth 가 0 근처 (manual GT 검증, ±5° 이내)
DEPTH_FRONT_SIGN = +1.0    # +1: front_center = centroid + R @ (0,0,+depth/2) — 검증 완료

# ===== camera intrinsic (RealSense D435i, header.txt 기준) =====
# 기존 main_rec.py 는 라이브 RealSense intrinsic 을 직접 사용.
# 시퀀스 재생 시 fallback:
CAMERA_FX = 614.18
CAMERA_FY = 614.31
CAMERA_CX = 329.28
CAMERA_CY = 234.53

# ===== INSERT_FORWARD (포켓 진입, fsm/align.py 의존) =====
# task.yaml: robot.fork.insertion_depth_m = 0.50 과 일치.
PALLET_POCKET_M = 0.5     # 정렬 후 추가 진입 깊이 (m)
INSERT_FWD_MPS  = 0.25    # 진입 평균 전진 속도 (m/s) — 실측 보정 권장
INS_FWD_MIN_SEC = 0.5
INS_FWD_MAX_SEC = 6.0     # 안전 클램프 (포크가 과하게 들어가는 것 방지)

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
# 6D 통합: detection 이 ok 이면 width 는 항상 PALLET_WIDTH_M 으로 채움.
# WIDTH_MIN_FULL = 0.00 이면 사실상 RECOVER 진입 안 함 (6D 의도와 일치).
WIDTH_MIN_FULL = 0.00  # m

# ===== ALIGN 단계: 거리 밴드 제어 =====
ALIGN_DIST_M = 2.20    # 정렬 목표 거리 (m)
ALIGN_BAND_M = 0.30    # 허용 밴드 (±m)

# ===== 정렬 완료 판정 안정화 =====
CMD_STABLE_THR = 5     # 같은 판정이 연속 n프레임 유지돼야 상태 전이

# ===== 회전 목표(각도 기반) =====
# 새 ALIGN 다이어그램은 타이머가 아닌 '상대 yaw 누적 90도 도달'로 회전을 종료합니다.
REL_YAW_TARGET_DEG = 85.0   # ALIGN_ROTATE_* → *_90 도달 조건

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

