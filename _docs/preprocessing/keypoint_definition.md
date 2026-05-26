# 키포인트 정의

> **컨벤션 결정 (2026-05-22)** — **Camera-facing convention 채택**.
> 0~3 = 카메라에 보이는 **가까운 면 (near face)**, 4~7 = 반대편 (far face).
> forklift 운용 시나리오 (카메라 = forklift 전방 = 항상 fork pocket 면 본다) 하에서
> 0~3 = fork pocket 면 자동 일치. 이전 object-fixed 정의는 **deprecated**.
> 결정 근거 + 영향: `project_keypoint_convention_decision.md` 메모리 + `_docs/history/2026-05-22.md`

## 1. 3D Cuboid Convention

팔레트를 감싸는 직육면체(cuboid)의 8개 꼭짓점 + 1개 centroid로 포즈를 표현한다.

### Y=UP Convention (3D 좌표계)

모든 3D 모델에서 **Y축이 위(UP)** 를 향하도록 정규화한다.

- `X` — 수평 폭 (medium, ~1.1m)
- `Y` — 높이 (height, ~0.11~0.15m) → **위 방향**
- `Z` — 수평 깊이 (depth, ~1.1~1.3m)

### Near / Far face 정의 (camera-facing)

- **Near face (= 0~3)**: 카메라에 가까운 면. 운용 시 = fork pocket 면 (forklift 가 fork pocket 면으로 접근하므로).
- **Far face (= 4~7)**: 반대편. 카메라에서 멀리.
- 어느 면이 near 인지 매 frame inference 시 결정 → post-process swap 으로 강제 (see `run_live.py`)

## 2. 키포인트 ID 매핑

```
     4 ──────── 5          ▲ y (UP)
    /|         /|          │
   / |        / |          └──── ▶ x
  0 ──────── 1  |         /  (camera direction)
  |  7 ──────|── 6       / z
  | /        | /         ▼
  |/         |/
  3 ──────── 2

  + centroid (8): 8개 꼭짓점의 평균
```

⚠️ 위 다이어그램에서 **z 축 = 카메라 방향 (camera looks toward -z)**.
   0~3 = z_max = 카메라에 가까운 near face.

### 꼭짓점 좌표 규칙 (cuboid local frame, Y=UP)

```
ID    X       Y       Z       위치 설명 (camera-facing)
──────────────────────────────────────────────────────────
0     min_x   max_y   max_z   near-top-LEFT     (가까운 면, 위, 좌)
1     max_x   max_y   max_z   near-top-RIGHT    (가까운 면, 위, 우)
2     max_x   min_y   max_z   near-bottom-RIGHT (가까운 면, 아래, 우)
3     min_x   min_y   max_z   near-bottom-LEFT  (가까운 면, 아래, 좌)
4     min_x   max_y   min_z   far-top-LEFT      (먼 면, 위, 좌)
5     max_x   max_y   min_z   far-top-RIGHT     (먼 면, 위, 우)
6     max_x   min_y   min_z   far-bottom-RIGHT  (먼 면, 아래, 우)
7     min_x   min_y   min_z   far-bottom-LEFT   (먼 면, 아래, 좌)
8     mid     mid     mid     centroid
```

### 면(Face) 정의

```
면                    꼭짓점          조건
─────────────────────────────────────────────────────────
Top (위)              {0, 1, 4, 5}   Y = max
Bottom (아래)         {2, 3, 6, 7}   Y = min
Near (카메라에 가까움) {0, 1, 2, 3}   Z = max  ★ camera-facing
Far (멀리)            {4, 5, 6, 7}   Z = min
Left                  {0, 3, 4, 7}   X = min
Right                 {1, 2, 5, 6}   X = max
```

### Forklift 응용 매핑 (운용 가정)

```
forklift 전방 카메라 → 항상 fork pocket 면 본다
                    → near face (0~3) = fork pocket 면
                    → pose R, t 출력 = fork pocket 방향
                    → forklift fork 진입 경로로 직접 사용
```

## 3. 변(Edge) 정의

[Geometric Filter (Step 2)](../method/step2_geometric_filter.md)에서 사용하는 12개 변:

```
방향          변 (꼭짓점 쌍)                물리 길이
──────────────────────────────────────────────────────
Width (X)     (0,1), (3,2), (4,5), (7,6)   ~1.1m
Height (Y)    (0,3), (1,2), (4,7), (5,6)   ~0.15m
Depth (Z)     (0,4), (1,5), (2,6), (3,7)   ~1.1m
```

### 대칭 쌍 (Flip Consistency용)

Filter A에서 좌우 flip 시 대응하는 키포인트 쌍:
- (0, 1), (3, 2), (4, 5), (7, 6)

## 4. Canonical Bbox 변환

각 USD 모델의 원본 좌표계를 Y=UP convention으로 정규화하는 과정:

```
R_canonical = R_yz_swap @ euler(base_rot)
```

- `euler(base_rot)`: 원본 bbox → X=medium, Y=long, Z=height
- `R_yz_swap = Rx(-90°)`: Y↔Z 스왑 → X=medium, Y=height(UP), Z=long(depth)

### ORIENTATION_OVERRIDES (검증 완료)

```
모델           base_rot        설명
──────────────────────────────────────────────────────────────
scene.usd      (180, 0, 90)    Z-thin, Rx(180°) top/bottom 교정 + Rz(90°)
scene_1.usd    (90, 0, 0)      Y-thin, Rx(90°)
scene_2.usd    (90, 0, 0)      Y-thin, Rx(90°)
scene_3.usd    (90, 0, 90)     Y-thin, Rz(90°)@Rx(90°)
```

검증 결과: edge 0→1 ≈ 1.0m (width), 0→3 ≈ 0.15m (height), 0→4 ≈ 1.2m (depth) — 모든 모델 일관.

## 5. 팔레트 규격

```
규격           가로(X)   세로(Z)   높이(Y)   비고
───────────────────────────────────────────────────
KS T-11형      1100mm    1100mm    150mm     본 연구 대상
EUR 팔레트     1200mm    800mm     144mm     참고용
```

> 규격 값은 `config/stage3_selftrain.yaml`의 `pallet` 섹션과 `config/default.yaml`의 `pallet` 섹션에서 관리.
