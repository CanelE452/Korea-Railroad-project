import os

# ============================================================
# 설정
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

GLTF_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "pallet", "pallet_scene", "Collected_World0", "SubUSDs", "textures",
)
GLTF_FILES = ["scene.gltf", "scene_1.gltf", "scene_2.gltf", "scene_3.gltf"]

USD_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "pallet", "models_usd")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "pallet", "training_data")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_LENGTH = 34.6    # D435i 기준 ~34.6mm (fx~615 at 640x480, sensor_width=36mm)
SENSOR_WIDTH = 36.0

PALLET_TARGET_SIZE = 1.2

# v11: 카메라 제약 — real 이미지 분석 기반 3모드 설계
# Real 카메라: RealSense D435i, 리프터 포크 마스트에 장착, 높이 ~0.3m
# Mode A (60%): 리프터 마운트 — real과 동일한 시점 (가장 중요)
# Mode B (25%): 약간 높은 시점 — 운전석/점검 시점
# Mode C (15%): 바닥 레벨 — 다양성 유지 (기존 전면 뷰)
CAMERA_MODE_PROBS = [0.60, 0.25, 0.15]  # A, B, C 확률

# Mode A: 리프터 마운트 시점 (real 이미지와 동일)
CAMERA_CONSTRAINTS = {
    "height_min": 0.2,     # m (포크 레벨 — real에서 관측)
    "height_max": 0.5,     # m (마스트 하단)
    "pitch_min": -5,       # degree (거의 수평)
    "pitch_max": 5,        # degree (약간 위를 올려봄)
    "distance_min": 1.5,   # m (접근 시작)
    "distance_max": 8.0,   # m (real에서 최대 관측 거리 — v11에서 확장)
    "yaw_min": -30,        # degree (좁은 범위 — 리프터는 정면 접근)
    "yaw_max": 30,         # degree
}

# Mode B: 높은 시점 (운전석/점검)
CAMERA_MODE_B = {
    "height_min": 0.5,     # m
    "height_max": 1.2,     # m
    "pitch_min": -25,      # degree (내려다봄)
    "pitch_max": 5,        # degree
    "distance_min": 1.0,   # m
    "distance_max": 6.0,   # m
    "yaw_min": -45,        # degree
    "yaw_max": 45,         # degree
}

LOOK_AT_TARGET = (0, 0, 0.08)
LOOK_AT_JITTER = 0.15

# v11: Mode C — 바닥 레벨 (다양성 유지, 기존 전면 뷰)
FRONT_VIEW_PROBABILITY = None  # v11에서 CAMERA_MODE_PROBS로 대체
FRONT_VIEW_CONSTRAINTS = {
    "height_min": 0.05,    # m (포크 최저 높이 — 순수 앞면)
    "height_max": 0.3,     # m (바닥 근접)
    "distance_min": 1.0,   # m
    "distance_max": 5.0,   # m
    "yaw_min": -20,        # degree (좁은 정면)
    "yaw_max": 20,         # degree
}

# v5: 디스트랙터 — 매 프레임 표시할 최대 수 / 미리 생성할 총 풀 크기
MAX_DISTRACTORS_PER_FRAME = 10
MAX_DISTRACTOR_POOL = 24   # 카테고리별로 3~4개씩 6카테고리 = 24개 풀

# v11: 팔레트 위 적재 — real에서 ~80%+ 적재물 관찰
#   20% 가림 없음, 40% 약한(1-20%), 25% 보통(20-40%), 15% 심한(40-50%)
CARGO_ON_PALLET_PROBABILITY = 0.8   # 80% 프레임에 적재물 있음
# 권장 가림 분포 (research guide v3 §3.7.5):
#   가림 없음 (0%):      40%  → CARGO_ON_PALLET_PROBABILITY=0.6이므로 40%는 카고 없음
#   약한 가림 (1~20%):   30%  → 0.50 × 0.6 = 30%
#   보통 가림 (20~40%):  20%  → 0.33 × 0.6 ≈ 20%
#   심한 가림 (40~50%):  10%  → 0.17 × 0.6 ≈ 10%
#   극단적 가림 (50%+):   0%  ← 생성하지 않음
CARGO_OCCLUSION_TIERS = [
    # (확률 within cargo, 최대 개수, 목표 크기 범위 m) — 확률 합 = 1.0
    # 목표 크기 = 카고의 max_dim이 이 범위(m)가 되도록 정규화 스케일 적용
    # 팔레트 상면(~1.1×1.2m) 기준
    # v11: real 이미지 기준 — 팔레트 위에 큰 박스가 쌓인 상태가 대부분
    (0.15, 2, (0.30, 0.50)),  # 약한 가림: 중간 물체 1~2개
    (0.35, 4, (0.40, 0.70)),  # 보통 가림: 큰 물체 2~4개
    (0.30, 6, (0.50, 0.80)),  # 심한 가림: 대형 물체 3~6개
    (0.20, 8, (0.60, 1.00)),  # 극심한 가림: 팔레트 덮는 대형 물체 4~8개
]
PALLET_SURFACE_HEIGHT = 0.15        # 팔레트 상면 높이 (m)
CARGO_Z_CLEARANCE = 0.01            # 팔레트 상면에 밀착 (m)

# Path Tracing 수렴 설정
WARMUP_STEPS = 2
RT_SUBFRAMES_PT = 2
RT_SUBFRAMES_RTL = 2

PALLET_COLORS = [
    (0.0, 0.2, 0.8),   # 파랑
    (0.8, 0.1, 0.1),   # 빨강
    (0.25, 0.25, 0.25), # 어두운 회색
    (0.45, 0.45, 0.45), # 회색
    (0.4, 0.25, 0.1),  # 갈색
    (0.1, 0.6, 0.2),   # 초록
    (0.8, 0.7, 0.1),   # 노랑
    (0.0, 0.5, 0.6),   # 청록
]

# 팔레트 tilt 없음 — 바닥에 수평 고정
PALLET_TILT_MAX = 0.0

# v4: Isaac Sim 내장 에셋 경로 (warehouse 환경 + Props)
WAREHOUSE_USD = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
DISTRACTOR_PROPS = [
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
]

# 로컬 Isaac Sim 에셋 (Nucleus 없이 사용)
ISAAC_ASSETS_ROOT = os.path.join(
    PROJECT_ROOT, "data", "pallet", "isaac_assets",
    "Assets", "Isaac", "4.5", "Isaac"
)

# v4: 배경 환경 다양화 — 4종 warehouse 변형을 매 프레임 랜덤 선택
WAREHOUSE_VARIANTS_LOCAL = [
    os.path.join(ISAAC_ASSETS_ROOT, "Environments", "Simple_Warehouse", f).replace("\\", "/")
    for f in [
        "warehouse.usd",
        "warehouse_multiple_shelves.usd",
        "warehouse_with_forklifts.usd",
    ]
]
WAREHOUSE_USD_LOCAL = WAREHOUSE_VARIANTS_LOCAL[0]  # 기본값 (호환성)

# v5: 디스트랙터를 카테고리별로 분류 — 균등 샘플링에 사용
DISTRACTOR_CATEGORIES = {
    "cardbox": [
        "Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxB_02.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxC_02.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxD_03.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
    ],
    "barrel": [
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_D_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BarelPlastic_D_02.usd",
    ],
    "crate": [
        "Environments/Simple_Warehouse/Props/SM_CratePlastic_A_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CratePlastic_B_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CratePlastic_C_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CratePlastic_D_01.usd",
        "Environments/Simple_Warehouse/Props/SM_CratePlastic_E_01.usd",
    ],
    "bottle": [
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticA_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticB_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticC_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticC_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticD_02.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
        "Environments/Simple_Warehouse/Props/SM_BottlePlasticE_02.usd",
    ],
    "sign": [
        "Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
        "Environments/Simple_Warehouse/Props/S_AisleSign.usd",
        "Environments/Simple_Warehouse/Props/SM_SignA_02.usd",
        "Environments/Simple_Warehouse/Props/SM_SignA_03.usd",
        "Environments/Simple_Warehouse/Props/SM_SignB_02.usd",
        "Environments/Simple_Warehouse/Props/SM_SignB_05.usd",
        "Environments/Simple_Warehouse/Props/SM_SignCVer_01.usd",
        "Environments/Simple_Warehouse/Props/SM_SignCVer_02.usd",
        "Environments/Simple_Warehouse/Props/SM_SignCVer_03.usd",
        "Environments/Simple_Warehouse/Props/SM_EmergencyBoardFull_01.usd",
    ],
    "misc": [
        "Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
        "Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
        "Environments/Simple_Warehouse/Props/SM_FireExtinguisher_02.usd",
        "Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "Environments/Simple_Warehouse/Props/SM_FuseBox_01.usd",
        "Environments/Simple_Warehouse/Props/SM_FuseBox_04.usd",
        "Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
        "Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
    ],
}
# flat list for loading (backward compat)
DISTRACTOR_PROPS_LOCAL = [p for cat in DISTRACTOR_CATEGORIES.values() for p in cat]

# v11: 배경 혼합 비율 — real은 야외 100%이므로 야외 비율 대폭 증가
WAREHOUSE_BG_PROBABILITY = 0.2   # 창고 20% (다양성 유지)
OUTDOOR_BG_PROBABILITY = 0.6    # 야외(HDRI) 60% (real 환경 반영)
HDRI_DIR_LOCAL = os.path.join(
    PROJECT_ROOT, "data", "pallet", "isaac_assets",
    "Assets", "Isaac", "4.5", "NVIDIA", "Assets", "Skies"
).replace("\\", "/")

# v3: 바닥 텍스처 다양성 (fallback용 OmniPBR diffuse 범위)
# warm/cool tone 변화 포함: R>B=warm(콘크리트), B>R=cool(에폭시)
FLOOR_DIFFUSE_RANGE = ((0.12, 0.11, 0.09), (0.60, 0.55, 0.53))
WALL_DIFFUSE_RANGE = ((0.18, 0.17, 0.15), (0.65, 0.62, 0.60))

# 프로시저럴 텍스처 저장 디렉토리
PROCEDURAL_TEX_DIR = os.path.join(PROJECT_ROOT, "data", "pallet", "_procedural_textures")
