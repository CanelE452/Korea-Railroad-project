# 7. 다양한 형태의 팔레트 일반화 전략

## 7.1 핵심 관찰

플라스틱 팔레트는 내부 슬롯 패턴, 다리 형태, 색상은 종류마다 다르지만,
**외곽 형상은 직육면체로 거의 동일하다.** 이것이 일반화가 가능한 근거이다.

## 7.2 일반화 4단계

```
전략 1: 다종 3D 모델 혼합 학습 (Step 1)
        → 공통 구조(직육면체 외곽)에 집중하도록 유도
        → Isaac Sim + Blender 1:1 혼합 (mixed_v8 = 9k, mixed_v10 = 10k)

전략 2: 극단적 재질/텍스처 Domain Randomization (Step 1)
        → 내부 디테일 의존성 제거, edge/corner 의존성 강화
        → 목재/플라스틱/금속 재질, 빛 7K~300K 랜덤, PT 렌더 등

전략 3: Soft-argmax coordinate Huber loss (Fine-tune stage)
        → belief MSE 만으로는 sub-pixel keypoint 위치가 1~2 px 밀림
        → coord Huber loss (Sun et al. 2018) 로 직접 좌표 supervision
        → noapril PnP rate 49.5% → 62.2% (v8_A, Loss ablation §11.3)
        → **keypoint 정밀도 향상 → RANSAC consensus 통과율 상승 → self-training 효율**
          이 chain 이 Loss / Filter / Self-training 을 하나의 story 로 잇는다

전략 4: Self-Training (Step 2~3) ← 핵심 Contribution
        → synthetic에서 학습 못한 다양한 형태에 자동 적응
        → RANSAC subset consensus 필터가 "검출된 keypoint 들이
          단일 6D pose 로 수렴하는가" 를 직접 검증 — 형태/규격에
          무관하게 pose consistency 만으로 pseudo-label 신뢰도 판정
        → 필터 채택률은 coord loss 가 강할수록 올라감 → 전략 3 과 상승작용

전략 5: 반복적 개선 (NVIDIA 공식 프로세스)
        → 실패 사례 분석 → 해당 변형 합성 데이터 추가 → 반복
        → v1 → v8 → v10 변천이 이 축의 실제 기록
```

### 전략 간 상호작용 요약

```
       [전략 1] 다종 3D 혼합 ─────────────┐
                                           │ belief map 학습
       [전략 2] DR ──────────────────────┤ (structure prior)
                                           │
                                           ▼
       [전략 3] coord ft ─────────► sharp keypoint
                                           │
                                           ▼
       [전략 4] RANSAC filter ─────► high PL quality
                                           │
                                           ▼
                                    Self-training
                                           │
                                           ▼
       [전략 5] 실패 분석 ────────► 합성 데이터 재생성
                 (루프)
```

## 7.3 실험 검증

```
Test Set:
  (a) Seen pallet:   학습에 사용한 3D 모델과 동일한 종류의 실제 팔레트
  (b) Unseen pallet: 학습에 사용하지 않은 종류의 플라스틱 팔레트

→ Unseen에서의 개선폭이 크면
  "self-training이 일반화 성능을 향상시킨다"는 강력한 주장 가능
```

---

# 8. 데이터셋 구성

> 합성 데이터 생성/검증/병합 파이프라인 → [data_pipeline.md](../preprocessing/data_pipeline.md) 참조.

### 8.1 현재 보유 데이터

```
데이터셋              출처                   라벨           용도                     수량
──────────────────────────────────────────────────────────────────────────────────────────────────
mixed_v8              Isaac Sim + Blender    자동 GT        Step 1 scratch 60 ep     9,000 장
mixed_v9              Isaac Sim + Blender    자동 GT        mid-term baseline        ~8,500 장
mixed_v10             Isaac Sim + Blender    자동 GT        **폐기** (annotation broken)  10,000 장
Synthetic Val         Isaac Sim (별도 seed)  자동 GT        PCK screening            ~1,000 장
capture0403noapril    직접 촬영              없음           Step 2~3 unlabeled pool  188 장
capture0403middle     직접 촬영              2D reproj GT   Filter P/R + 모델 평가   440 장
```

### 8.2 촬영 대기 (논문 Tier 1 필수)

```
데이터셋              출처                   라벨           용도                     상태
──────────────────────────────────────────────────────────────────────────────────────────────
Real Test — Seen      직접 촬영 + AprilTag   GT 예정        Seen 평가 (§11.6)         TBD
Real Test — Unseen    직접 촬영 + AprilTag   GT 예정        Unseen 일반화 평가       TBD
Real Test — Wood      직접 촬영 + AprilTag   GT 예정        도메인 외삽 (future)     TBD
```

### 8.3 capture0403middle limitation

capture0403middle 은 2D projected cuboid GT 만 신뢰 가능. 3D ADD 는 object
frame convention mismatch 로 사용 불가. 따라서:

- **Primary metric**: 2D mean projected_cuboid reproj error (threshold 50 px)
- **3D 수치 금지**: 논문 본문 표 / figure 어디에도 이 데이터셋의 ADD / 5 cm 5° 를 쓰지 않는다
- **AprilTag GT 가 해결되기 전까지는 Seen/Unseen 3D 평가를 이 데이터셋으로 할 수 없음** — 촬영이 필요
- **현재 용도**: (a) 필터 후보 P/R screening (b) 모델 상대 순위 (c) 정성 failure mode

필터 선정 (2026-04-11) 에 사용된 데이터셋. 상세: `_docs/filter/2026-04-11_selection.md` §2.
