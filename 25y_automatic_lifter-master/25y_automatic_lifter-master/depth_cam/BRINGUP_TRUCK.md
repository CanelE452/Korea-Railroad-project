# 트럭 적재 (Phase B) 브링업 가이드

파렛트 삽입 → 트럭 적재 전 과정을 새 CAN 코어 위에서 오류 없이 돌리기 위한
현장 검증 절차. **순서를 건너뛰지 말 것** — 각 단계가 다음 단계의 전제를 확정한다.

## 0. 코드 구조 (2026-07 정리)

```
depth_cam/calib/can/          ← CAN 단일 진실 (기존 4벌 중복 통합)
  protocol.py                 ID/템플릿/현장 플래그 (CAN_ID_FAMILY, LIFT_CODES_SWAPPED)
  bus.py                      TX 스레드: movement 50ms 재송신, 워치독, 재초기화, JSONL 로그
  commands.py                 주행(기존 바이트 동일) + 리프트/리치/폴드 모드 시퀀스
depth_cam/calib/fsm/
  mission.py                  PHASE_A(파렛트) → LIFT_PALLET → PHASE_B(트럭) → DONE
  truck.py                    Phase B FSM (다이어그램 T0~T27)
depth_cam/calib/truck/
  lasers.py                   TFmini-S 듀얼 리더 + 모서리/안착 감지기
  truck_adapter.py            SMOKE Detection → (ψ, d_lat, d_fwd)
  smoke_source.py             truck_loading/ SMOKE 번들 래퍼
depth_cam/truck_main.py       Phase B 단독 실행 (--dry-run 지원)
depth_cam/tools/bench_*.py    아래 브링업 스크립트
depth_cam/logs/can_*.jsonl    전 프레임 송신 로그 (사후 분석)
```

기존 파일 호환: `calib/control.py` 는 shim — `main_rec.py`/`fsm` 의 import 는 그대로 동작.
구 구현 (`CAN/control_forklift*.py`, `forklift_ctrl/`) 은 참고용 deprecated.

## 1. 사전 확인 (사무실, 하드웨어 불필요)

```bash
cd 25y_automatic_lifter-master/25y_automatic_lifter-master
python -m pytest tests/ depth_cam/tests_can/ -q     # 62 passed 여야 함
```

- 골든 테스트가 주행 프레임 바이트 회귀를 잡는다. 리팩터/수정 후 항상 실행.
- Phase B 로직 확인: `tests_can/test_truck_machine_e2e.py` (T0→T27 시뮬레이션 완주).

## 2. 현장 벤치 체크리스트 (실차, 순서 고정)

각 단계는 `depth_cam/tools/` 스크립트. 모든 송신 프레임이 `logs/can_*.jsonl` 에 남는다.

| # | 스크립트 | 확인 사항 | 실패 시 |
|---|---|---|---|
| 1 | `bench_heartbeat.py` | 버스 에러 0건 (CanKing/candump) | 채널/비트레이트 점검. 차량 무반응 → `CAN_ID_FAMILY=E4` 재시험 |
| 2 | `bench_mode.py` | driving↔lift↔reach↔folding 모드 표시 전환 | settle 늘리기 (`config.MODE_SWITCH_SETTLE_S`) |
| 3 | `bench_lift_jog.py` | **LIFT_UP 펄스에 포크가 올라가는가** | 내려가면 `LIFT_CODES_SWAPPED=1` 재시험 → 맞으면 protocol.py 기본값 수정 |
| 4 | `bench_drive_jog.py` | 전/후진/회전/저속전진 1회씩 부드럽게 | 재송신 관련 이상 시 `bus.py MOV_RESEND_PERIOD` 조정 |
| 5 | (수동) 리프트 속도 캘리브레이션 | **적재 상태**로 0.5 m 상승 시간 실측 | `config.LIFT_SPEED_MPS` 갱신 (하강은 `LOWER_SPEED_MPS`) |
| 6 | `main_rec.py` (기존 파렛트 시퀀스) | 새 CAN 코어로 파렛트 정렬~삽입 회귀 통과 | **통과 전 Phase B 진행 금지** |
| 7 | `truck_main.py --dry-run` | 레이저 L/R 값 정상 수신 (배선 확정) | `LASER_WIRING=single_port\|dual_port`, `LASER_PORT*` 환경변수 조정 |
| 8 | `truck_main.py` (블록 위 드라이런 → 저속 실전) | T0→T27 상태 궤적 | E-stop 대기 인원 필수 |

### 레이저 배선 확정 (7단계)

레이저 2개(좌/우 포크)가 어느 형태인지 미확정 — 코드는 두 모드 지원:

- **한 시리얼에 두 채널** (아두이노가 `L1 70cm strength=...` / `L2 ...` 송신):
  `LASER_WIRING=single_port LASER_PORT=COM6` (기본값)
- **COM 포트 2개**: `LASER_WIRING=dual_port LASER_PORT_L=COM6 LASER_PORT_R=COM7`

`--dry-run` 화면에서 L/R 값이 모두 갱신되면 확정. 왼쪽 포크를 손으로 가려
L 만 변하는지(채널 매핑 방향) 반드시 확인 — 뒤집혔으면 `LASER_CH_L/R` 교체.

### 캘리브레이션 대상 상수 (`calib/config.py`)

| 상수 | 의미 | 확정 방법 |
|---|---|---|
| `LIFT_SPEED_MPS` | 리프트 속도 | 벤치 5단계 실측 (적재 상태) |
| `SLOW_FWD_MPS` | forward_slow 속도 | 벤치 4단계 실측 |
| `LOWER_SPEED_MPS` | 하강 속도 | 벤치 5단계 실측 |
| `LASER_DROP_THRESH_M` | 모서리 급감 임계 (기본 0.30) | 드라이런에서 바닥↔적재면 차 실측의 ~50% |
| `CAM2_TO_FORK_T/RPY` | Camera2 extrinsic | 자/각도기 실측 (포크 초기 높이 기준) |
| `TRUCK_YAW_TOL_DEG`, `TRUCK_OFF_TOL_M` | 접근 허용치 | 드라이런 후 조정 |

## 3. 실행 형태

- **전체 미션** (파렛트→트럭, 다이어그램 P15⇒T0):
  `python main_rec.py --truck-after-done [--truck-camera-id N]`
- **트럭 단계만** (현장 반복 테스트): `python truck_main.py [--dry-run]`

## 4. 안전 장치 (자동)

- **movement 재송신 (50ms)**: 프레임 유실돼도 명령 유지 — 구 fire-once 제거
- **워치독 (1.0s)**: FSM 루프가 죽으면 TX 스레드가 정지 프레임만 송신
- **버스 재초기화**: TX 연속 실패 시 busOff/busOn 백오프 재시도, 그동안 FSM 은 STOP
- **FAULT 상태**: 레이저 stale / 모서리 탐색·하강 타임아웃 → 즉시 STOP 유지 (수동 개입)
- **STOP 인터록 (1.2s)**: 모든 명령 전환 사이 (파렛트 FSM 과 동일 규약)

## 5. 문제 발생 시

1. `depth_cam/logs/can_*.jsonl` 에서 마지막 프레임들의 `src` 태그 확인
   (어느 상태가 어떤 프레임을 보냈는지 추적 가능)
2. `python -m pytest depth_cam/tests_can/ -q` 로 로직 회귀인지 하드웨어인지 분리
3. 골든과 실차 반응이 다른 프레임은 `tests_can/golden/driving_commands.json` 과 대조
