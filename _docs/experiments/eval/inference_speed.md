# Inference Speed Breakdown

상태: **★ 완료** (2026-04-13)

## 환경

```
GPU:       NVIDIA GeForce RTX 3080
CUDA:      12.6
CPU:       (system default)
Input:     640x480 → 448x448 resize
Model:     DOPE (VGG-19 backbone, fp32)
Batch:     1 (실시간 inference 가정)
Warmup:    5 iterations, 측정: 100 iterations 평균
```

## 결과

```
Stage                        Time (ms)     %
─────────────────────────────────────────────
Preprocessing (resize+norm)     8.80     23.5%
DOPE forward pass              22.52     60.2%
Keypoint extraction             1.52      4.1%
PnP solve (EPnP+RANSAC)        4.54     12.2%
─────────────────────────────────────────────
Total                          37.39    100.0%
FPS                             26.7
```

## 해석

- **DOPE forward pass 가 60% 지배** — VGG-19 backbone (6-stage belief + affinity head) 이 병목
- **Keypoint extraction 4%** — gaussian filter + peak detection, negligible
- **PnP solve 12%** — EPnP + RANSAC, CPU-bound
- **Preprocessing 24%** — cv2.resize + normalize, CPU-bound
- **26.7 FPS** — 실시간 가능 (30 FPS 에 근접)

## 최적화 가능성 (미실행)

- TensorRT / ONNX export: forward pass 22ms → ~8ms 예상 (3x speedup)
- Preprocessing GPU: torchvision.transforms.v2 GPU resize → 8ms → ~1ms
- 총 최적화 후: ~15ms, ~67 FPS 기대

## 논문 draft 한 줄

> "On an RTX 3080, the full pipeline (DOPE forward 22.5 ms + keypoint
> extraction 1.5 ms + PnP 4.5 ms) runs at **26.7 FPS**, demonstrating
> real-time feasibility for industrial deployment."

## 관련

- 모델: `weights/v8_ablation_A_coord/final_net_epoch_0065.pth`
- 추론 코드: `scripts/data_prep/visualize_inference.py`
- PnP: `scripts/self_training/pnp_solver.py`
- 실시간 추론 구현: `scripts/dope/run_dope_live.py`
