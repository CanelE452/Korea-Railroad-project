# Method Overview — 논문용 일반화 파이프라인 (camera-facing)

> 2026-06-04 재작성. 폐기된 v8(object-frame) 설계는 `archive/overview.md` 참조.
> memory `two-tracks-paper-vs-challenge`, `camera-facing-0123-convention`.

## 문제 정의

처음 보는 파렛트(비율·외형·조명 제각각)의 6D pose(=9 keypoint)를 추정한다.
**내 실제 파렛트(v1/v2)는 학습에 쓰지 않고**, 인터넷 무료 3D 팔레트 모델 기반
합성 데이터만으로 학습해 **일반화**를 달성하는 것이 논문 핵심.

## 두 트랙 (구분 필수)

| | 논문용 (`paper_*`) | 과제용 (`challenge*`) |
|---|---|---|
| 목표 | 처음 본 파렛트 일반화 | 내 파렛트 과적합, forklift 배포 |
| 데이터 | v1/v2 제외, 합성(mixed_v8) | v1/v2 포함 |
| convention | camera-facing 0123 | camera-facing 0123 |

본 문서는 **논문용** 파이프라인.

## 파이프라인

```
Step 1  합성 데이터 + DOPE 학습
        - camera-facing 0123 합성 (mixed_v8, v1/v2 제외)
        - squash: 여러 aspect ratio 로 비율 강건성 (처음 본 파렛트 비율 대응)
        - truncation padding: 잘린 코너의 belief 를 padding 영역에 supervise
        → paper_base

Step 2  기하 필터 + Pseudo-label
        - paper_base 로 unlabeled 추론 → 9 keypoint 예측
        - 2D projective 기하 필터로 신뢰도 높은 PL 만 선별 (PnP 불필요)
        → 처음 본 파렛트(비율 unknown)도 필터링 가능

Step 3  Self-training Finetuning (반복)
        - 선별된 PL 로 finetune → paper_r1 → PL 재추출 → paper_r2 ...
        - 핵심 교훈: PL 수보다 품질(신뢰도). 좋은 필터가 성공 열쇠.
```

## 핵심 설계 결정

- **convention = camera-facing 0123** → 직사각형 2D 기하 필터 가능 (이전 object-frame 에선 불가)
- **PnP 용도 분리**: 필터 = 2D 기하(PnP 불필요) / 평가·거리 = SQPnP(치수 known 데이터)
- **비율 강건성 = squash** (고정 치수 가정 폐기 → 일반화)
- **truncation = padding** (잘린 이미지 강건)

## 관련 문서

- Step 1: `step1_synthetic_data.md`
- Step 2: `step2_geometric_filter.md`
- Step 3: `step3_selftraining.md`
- 평가: `evaluation.md`
- keypoint: `../preprocessing/keypoint_definition.md`
- 모델: `../models/paper_base.md`
