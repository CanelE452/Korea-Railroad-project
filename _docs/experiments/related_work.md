# T10. Related Work 비교 표

상태: **예정 (Phase 5, 논문 draft)**

## 목적

본 연구의 contribution 차별성을 관련 연구들과 수치로 대비.

## Table 10 (draft)

```
Method                  Sensor     Synth Size    Real Adaptation      Eval Environment         Metric
─────────────────────────────────────────────────────────────────────────────────────────────────────────
DOPE (2018)             RGB        ~60K          ✗                    lab + YCB                 ADD / ADD-S
Knitt et al. (2022)     RGB        ~50K          ✗                    lab, single pallet        2D IoU / custom
Mueller et al. (2023)   RGB        ~30K          ✗                    synthetic + simple real   PCK / reproj
UDA-COPE (2022)         RGB-D      —             ✅ (ST + depth)       category-level NOCS       NOCS mAP
PseudoFlow (2023)       RGB        YCB / LM      ✅ (optical flow ST)  YCB-V / LM-O              ADD-S
Ours (2026)             RGB        ~10K          ✅ (RANSAC filter ST) real warehouse + DR       PnP% / ADD / 5 cm 5°
```

## Contribution 차별화 포인트

1. **Real self-training with a geometric (not flow / depth) filter**
   - UDA-COPE 는 depth 의존, PseudoFlow 는 영상 광류 의존
   - Ours 는 단일 RGB frame + RANSAC subset consensus 만 사용 → 센서 / 데이터
     요구 최소

2. **더 작은 합성 데이터셋으로 동등 또는 더 좋은 real 성능**
   - DOPE / Knitt / Mueller 대비 1/3 ~ 1/5 크기의 합성 데이터
   - Isaac + Blender 혼합으로 domain gap 완화

3. **엄밀한 필터 P/R 비교를 contribution 자체로 승격**
   - 23 필터 후보 GT 기반 비교 ([`filter/selection.md`](./filter/selection.md))
   - canonical filter 의 negative result 도 contribution
   - Related work 에서는 "필터를 고른 근거" 를 이 수준으로 제시한 논문 없음

4. **Industrial domain (plastic pallet in warehouse)**
   - YCB / LM / NOCS 는 tabletop 객체 위주
   - Knitt / Mueller 는 pallet 이지만 lab 환경 / 단순 배경

## 논문 draft 문구

> "Unlike prior pseudo-label self-training methods that rely on depth
> supervision (UDA-COPE) or temporal optical flow (PseudoFlow), our
> pipeline uses only RGB frames and validates pseudo-labels via RANSAC
> subset consensus — a single-frame geometric filter selected via
> a rigorous 23-candidate precision/recall comparison. With a third to
> a fifth of the synthetic data of prior industrial pallet methods
> (~10K images vs 30K–50K), we achieve pose estimation in a real
> warehouse environment, on pallet types unseen during training."

## 확장 검토 대상 (optional)

```
- Self6D (2020)           : RGB self-supervised, no-SSS
- GDR-Net (2021)          : synth-to-real direct
- RKHSPose (2024)         : latest self-supervised 6DoF
- 3DUDA (2024)            : source-free category UDA
```

필요 시 Appendix 로 확장.

## 관련

- Survey: `_docs/survey/survey-6d-pose-estimation.md`
- Implementation contribution: `_docs/method/implementation.md` §13
