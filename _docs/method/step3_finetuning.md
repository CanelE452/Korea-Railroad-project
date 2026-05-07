# Step 3: Finetuning + 반복적 Self-Training

## 5.1 학습 데이터

```
학습 데이터 = Synthetic (labeled) + Real (pseudo-labeled)

Synthetic: Isaac Sim에서 생성된 원래 학습 데이터 (GT label)
Real:      Step 2에서 Geo Filter를 통과한 pseudo-label
```

## 5.2 Loss 구성

```python
def compute_finetuning_loss(model, batch_syn, batch_real, alpha=1.0):
    # Synthetic loss (GT 기반)
    pred_syn = model(batch_syn["image"])
    L_pose_syn = F.mse_loss(pred_syn["belief"], batch_syn["gt_belief"]) \
               + F.mse_loss(pred_syn["affinity"], batch_syn["gt_affinity"])

    # Real loss (pseudo-label 기반, strong augmentation 적용)
    image_strong = apply_strong_augmentation(batch_real["image"])
    pred_real = model(image_strong)
    L_pose_real = F.mse_loss(pred_real["belief"], batch_real["pseudo_belief"])

    # Total
    L_total = L_pose_syn + alpha * L_pose_real
    return L_total
```

## 5.3 Strong Augmentation (Real 이미지에 적용)

```python
strong_augmentation = {
    "color_jitter": {"brightness": 0.4, "contrast": 0.4,
                     "saturation": 0.4, "hue": 0.1},
    "random_erasing": {"probability": 0.5, "scale": "(0.02, 0.2)"},
    "gaussian_blur": {"probability": 0.5, "kernel_size": "(3, 7)"},
    "gaussian_noise": "N(0, 0.05)",
}
# 색상(photometric) 변환만 사용
# 기하학적 변환은 keypoint 좌표 변환이 필요하므로 미사용
```

## 5.4 학습 설정

```yaml
Step3_Finetuning:
  optimizer: Adam
  learning_rate: 1e-5       # Step 1보다 낮게 (finetuning이니까)
  batch_size: 8
  epochs_per_round: 3~5
  alpha: 1.0                # pseudo-label loss 가중치
  synthetic_ratio: 0.5      # 배치 내 synthetic 비율
  real_ratio: 0.5           # 배치 내 pseudo-labeled real 비율
```

---

# 6. 반복 (Iterative Self-Training)

## 6.1 반복 구조

```
Round 1: Step 1 모델 → Step 2 (엄격한 필터) → Step 3 finetuning
         → 소수의 확실한 pseudo-label로 학습

Round 2: Round 1 모델 → Step 2 → Step 3
         → 모델이 좋아졌으니 pseudo-label 더 많이 통과

Round 3: Round 2 모델 → Step 2 → Step 3
         → 추가 개선, 수렴 확인

수렴 기준: pseudo-label 채택률 변화 < 1% for 연속 2 라운드
보통 2~3 라운드면 수렴
```

## 6.2 Self-Training 전체 루프

```python
def self_training(model, synthetic_loader, real_unlabeled_loader,
                  pallet_3d_kp, camera_matrix,
                  num_rounds=3, epochs_per_round=5):

    for round_idx in range(num_rounds):
        print(f"=== Round {round_idx + 1} ===")

        # ---- Step 2: Pseudo-label 생성 + 필터링 ----
        model.eval()
        pseudo_data = []

        with torch.no_grad():
            for image in real_unlabeled_loader:
                belief_maps = model(image)
                keypoints_2d = extract_peaks(belief_maps)

                is_valid, reason = geometric_filter(
                    model, image, keypoints_2d,
                    pallet_3d_kp, camera_matrix
                )
                if is_valid:
                    pseudo_data.append({
                        "image": image,
                        "pseudo_belief": belief_maps,
                    })

        acceptance_rate = len(pseudo_data) / len(real_unlabeled_loader)
        print(f"Pseudo-label 채택률: {acceptance_rate:.1%}")

        # ---- Step 3: Finetuning ----
        model.train()
        for epoch in range(epochs_per_round):
            for batch_syn, batch_real in zip_longest(
                synthetic_loader, pseudo_data_loader(pseudo_data)
            ):
                loss = compute_finetuning_loss(model, batch_syn, batch_real)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    return model
```
