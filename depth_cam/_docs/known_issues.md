# Known Issues — 복사 직후 식별된 문제

원본 `25y_automatic_lifter-master/depth_cam`을 그대로 가져왔을 때 발견된 문제들. 통합 전에 처리하지 않으면 `python main_rec.py` 실행 시 즉시 에러가 난다.

## I-1. `ui.diagram` 모듈 누락 (치명적)

`main_rec.py:23`:
```python
from ui.diagram import draw_fsm_diagram_panel
```

- **현상**: `depth_cam/ui/` 폴더 자체가 없음. 원본 repo에도 없음 (`25y_automatic_lifter-master`에서 search 결과 0건). `main_rec.py` 임포트 단계에서 `ModuleNotFoundError`.
- **사용처**: `main_rec.py:199, 342` — FSM 상태 다이어그램을 우측 패널로 hconcat.
- **해결안 (택1)**:
  - **(A) stub 모듈 생성**: `depth_cam/ui/__init__.py` + `depth_cam/ui/diagram.py`를 만들어, `draw_fsm_diagram_panel(fsm, panel_size=(h, w))`가 그냥 검은 패널 + FSM 현재 상태 문자열만 작은 텍스트로 표시하도록.
  - **(B) import + hconcat 제거**: `main_rec.py`에서 `from ui.diagram import ...` 삭제, `diag` 관련 코드 삭제, `show = vis` 로 단순화.
- **권장**: (A). 시연 영상에서 FSM 상태가 직관적으로 보이는 게 좋음.

## I-2. `calib/fsm.py` ↔ `calib/fsm/` 동시 존재 (dead code 위험)

- **현상**: `calib/fsm.py` (단일 파일, 27838 bytes) 와 `calib/fsm/` (패키지, `__init__.py + top.py + align.py + ...`) 가 같은 디렉토리에 있음.
- **Python import 규칙**: 패키지(폴더)가 우선됨. `from calib.fsm import CalibrationFSM`은 `calib/fsm/__init__.py → from .top import CalibrationFSM` 로 해석. **즉 `calib/fsm.py`는 실행되지 않는 dead code**.
- **위험**: 누군가 `calib/fsm.py`를 수정하면 의도한 동작이 안 되어 디버깅에 시간 낭비할 수 있음.
- **해결안**: `calib/fsm.py` → `calib/fsm_legacy.py.bak`로 rename. 또는 삭제 후 git에서 복원 가능하도록 commit 해두기.

## I-3. INSERT_FORWARD 관련 config 누락

`calib/fsm/align.py:_fwd_sec_for_insertion`:
```python
pocket_m = float(getattr(cfg, "PALLET_POCKET_M", 0.0))   # default 0.0
v_mps    = float(getattr(cfg, "INSERT_FWD_MPS", 0.25))
t_min    = float(getattr(cfg, "INS_FWD_MIN_SEC", 0.5))
t_max    = float(getattr(cfg, "INS_FWD_MAX_SEC", 10.0))
```

- **현상**: 위 상수들이 `calib/config.py`에 정의되어 있지 않음. `getattr(default)` 덕분에 ImportError는 안 나지만, **PALLET_POCKET_M=0.0이면 dist_z만큼만 전진**하고 포크가 팔레트 안으로 충분히 들어가지 않음.
- **해결안**: `calib/config.py`에 다음 추가:
  ```python
  # ===== 포켓 삽입 (INSERT_FORWARD) =====
  PALLET_POCKET_M = 1.0      # 정렬 후 추가 진입 깊이 (m). KS T-11 양면형 포크 구멍 깊이 기준
  INSERT_FWD_MPS  = 0.25     # 평균 전진 속도 (m/s) — 실측 보정
  INS_FWD_MIN_SEC = 0.5      # 안전 클램프 (최소)
  INS_FWD_MAX_SEC = 10.0     # 안전 클램프 (최대)
  ```

## I-4. `MODEL_PATH` 가 YOLO 가중치 경로

`calib/config.py:5`:
```python
MODEL_PATH = 'runs/segment/y11n_seg_finetune/weights/last.pt'
```

- **현상**: 6D 통합 후 더 이상 YOLO 모델을 안 쓰는데 경로가 남아있음. 게다가 이 경로(`runs/segment/...`)는 원본 repo 기준이고 현재 가져온 폴더에는 `runs/detect/train/...`만 있음 → YOLO 그대로 쓰려고 해도 경로 mismatch.
- **해결안**: `MODEL_PATH` 를 DOPE 가중치 절대경로로 교체:
  ```python
  MODEL_PATH = r"C:/Users/minjae/Documents/github/FoundationPose/challenge/weights/baseline_v8_A.pth"
  ```
  또는 DOPE 전용 상수 `DOPE_WEIGHTS_PATH`로 분리하고 `MODEL_PATH`는 deprecated 처리.

## I-5. `requirements.txt` 에 `ultralytics`만 있고 `pyrr` 없음

DOPE 코드(`Deep_Object_Pose/common/cuboid_pnp_solver.py`)는 `pyrr.Quaternion`을 사용. 현재 `pallet-pose` env에는 이미 설치되어 있지만, depth_cam을 단독으로 setup하려면 누락.

- **해결안**: `requirements.txt` 수정:
  - `ultralytics==8.3.191` → 제거
  - `pyrr` → 추가 (FoundationPose 메인 requirements 확인 후 동일 버전)

## I-6. `runs/` 폴더 149MB — git 추적 시 문제

YOLO 학습 산출물(`runs/detect/train/weights/best.pt`, `last.pt`). 6D 통합 후 사용 안 함.

- **해결안**: 
  1. 일단 로컬에는 보존 (혹시 비교/롤백 필요할까봐)
  2. `.gitignore`에 `depth_cam/runs/` 추가
  3. 또는 즉시 삭제

## I-7. `logger.py` 의 import 경로 가정 충돌

`logger.py:275`:
```python
for name in ("calib.control", "depth_cam_2.calib.control"):
```

- **현상**: `depth_cam_2.calib.control`를 우선시도하지만, 현재 폴더명은 `depth_cam`. 이슈는 아니지만(첫 후보 `calib.control`이 succeed), 일관성 측면에서 정리할 만.
- **영향**: 없음. 그대로 둬도 동작.

## I-8. `MIN_POINTS = 120` 같은 RANSAC 관련 상수가 config에 남음

6D 통합 후 사용 안 함:
- `SAMPLE_STRIDE, Z_INLIER_THRESH, MIN_POINTS, PLANE_INLIER_THRESH, PLANE_MAX_TRIALS, EMA_ALPHA_WIDTH`

- **해결안**: 보존 (`geometry.py`가 reference로 남아 있으니 함수와 함께). 단 `_docs/`에 deprecated 표시.

---

## 우선순위 (통합 작업 전 처리할 순서)

1. **I-1 (ui.diagram 누락)** — main_rec.py가 import 단계에서 죽음. 가장 먼저.
2. **I-4 (MODEL_PATH)** — perception 교체와 함께 처리.
3. **I-3 (INSERT 관련 config)** — INSERT_FORWARD가 의도대로 작동하려면 필수.
4. **I-2 (fsm.py 정리)** — 안 해도 동작은 함. 단 혼동 방지 차원.
5. **I-5 (requirements)** — pallet-pose env에서 이미 충족. 다른 머신 setup 시에만.
6. **I-6 (runs/)** — git commit 직전.
7. **I-7, I-8** — 정리성 작업. 통합 후 여유 있을 때.
