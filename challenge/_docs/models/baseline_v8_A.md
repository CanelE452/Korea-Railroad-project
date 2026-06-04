# baseline_v8_A

## 학습 설정

```
Weight:      challenge/weights/baseline_v8_A.pth
Header:      challenge/weights/baseline_v8_A.header.txt
초기 weight: weights/v9_ablation_A_coord/final_net_epoch_0065.pth
Epochs:      68 (65 → 68, 즉 3 ep ft)
Batch size:  4
LR:          1e-5
Sigma:       4.0
Image size:  448
Seed:        7963
```

## 학습 데이터

```
data/pallet/training_data/mixed_v8_train     (Isaac Sim 합성, mixed_v8)
```

## Loss 설정

```
symmetric_loss : False
struct_loss    : True
  struct_lambda  : 1.0
  struct_coord   : 0.003
  struct_edge    : 0.0
  struct_flip    : 0.0
  struct_delta   : 0.03
  struct_warmup  : 0
geo_loss       : False
rel_loss       : False
```

## 메모

- v8 ablation series 의 공정 비교 baseline (`_docs/models/v8_ablation.md` 의 A_coord 계열)
- depth_cam 통합 시 `MODEL_PATH` 기본값 (`depth_cam/calib/config.py`)
- twin_pnp_check 50/50 frame 에서 contract `default_z180` + dim `(1.0, 1.2, 0.15)` 으로 reproj 2.89px, |dt|=0.085m 확인
- manual GT 27 frame (`capturepallet07_manual_gt`) 분석에서 정렬된 frame yaw 식 A (`atan2(front[0], -front[2])`) = +4.14° 로 거의 0, `YAW_OFFSET_DEG=0` 확정
- depth_cam gate 완화 (min_kp 7→5, z_max 5→7, depth_pnp_rel 0.30→0.50) 후 capturepallet03 detection 1/13 → 7/13 (54%)
