# Challenge — Forklift Pallet Insertion

이 폴더는 **연구용 코드(`scripts/`, `Deep_Object_Pose/`, `_docs/`)와 분리된 과제 전용 워크스페이스**다.

- 목적: 파렛트를 실시간 추론해서 리프터가 포크를 넣고 트럭에 싣는 데모.
- 정책: 과제용으로 자유롭게 overtune. 메인 연구 결과(P/R, ablation 등)와 비교 가치는 없다.
- baseline: `weights/v8_A_control/final_net_epoch_0068.pth` 스냅샷 (header 메타는 `weights/baseline_v8_A.header.txt`).

## 디렉토리

```
challenge/
├── README.md               이 파일
├── config/task.yaml        baseline meta + 추론 threshold + finetune + robot fork 설정
├── weights/
│   ├── baseline_v8_A.pth         v8_A_control/final_net_epoch_0068.pth 스냅샷 (192MB)
│   ├── baseline_v8_A.header.txt  학습 시 사용한 argparse Namespace
│   └── finetuned/                challenge 데이터로 ft 산출물 (생성 시)
├── scripts/
│   ├── run_live.py         RealSense 실시간 추론 (false positive gate 강화)
│   └── finetune.sh         challenge/data로 추가 ft (메인 train_dope.sh 위임)
├── robot/
│   └── fork_target.py      6D pose → fork entry pose 변환 스켈레톤
├── data/                   challenge 환경에서 수집한 RGB + GT (gitignore 일부)
└── _docs/                  과제 진행 기록 (필요 시)
```

## 실행

### 실시간 추론

```bash
conda activate pallet-pose
# (a) RealSense D435i 라이브
python challenge/scripts/run_live.py --realsense

# (b) 저장된 시퀀스 재생 (data/outside/capture* — RGB+Depth 페어)
python challenge/scripts/run_live.py --seq data/outside/capturepallet02
python challenge/scripts/run_live.py --seq data/outside/capturepallet07 --seq_fps 10
python challenge/scripts/run_live.py --seq data/outside/capturepallet11 --seq_loop

# (c) 일반 웹캠 (depth 없음 → depth-PnP gate 비활성)
python challenge/scripts/run_live.py --cam_id 0
```

기본 weight는 `config/task.yaml`의 `baseline.weights`. 다른 ckpt 사용 시 `--weights <path>`.

**시퀀스 모드 키**: `space`=일시정지, `n`=다음 프레임 (paused), `p`=이전 프레임, `q`=종료.
시퀀스의 `cam_K.txt`가 있으면 자동으로 K로 사용하고, 없으면 task.yaml의 camera intrinsic을 쓴다.

### 추가 finetune (데이터 수집 후)

```bash
# challenge/data/{train,val}/ 에 NDDS 포맷 ({i:06d}.png + {i:06d}.json) 채운 뒤
bash challenge/scripts/finetune.sh
```

### Fork target 계산 (smoke)

```bash
python challenge/robot/fork_target.py --smoke
```

## False positive 대책 (run_live.py)

메인 `scripts/dope/run_dope_live.py`는 기본 threshold가 매우 낮아 매 프레임 잘못된 detection이 발생한다.
challenge 버전은 다음 gate를 추가한다 (`config/task.yaml` → `inference.gates`):

```
- min_detected_keypoints: 7        (9개 중 7개 이상 검출되어야 인정)
- max_reproj_error_px:    8.0      (PnP reprojection error 상한)
- z_min_m / z_max_m:      0.30 / 5.00
- depth_pnp_z_max_rel:    0.30     (RealSense depth 와 PnP z 일치)
- cuboid_edge_ratio_tol:  0.30     (KS T-11 종횡비 sanity)
- temporal.confirm_frames: 2       (연속 2프레임 통과 시 CONFIRMED)
```

화면에는 detection 상태가 다음 셋 중 하나로 표시된다.

- `NOT DETECTED (reason)` — 어떤 gate에서 떨어졌는지 표시
- `PENDING n/N` — 단일 프레임은 통과했으나 temporal 미달
- `CONFIRMED` — gate + temporal 모두 통과

슬라이더로 `threshold / thresh_map / thresh_pts / min_kp / max_reproj_px`를 실시간 튜닝 가능.
