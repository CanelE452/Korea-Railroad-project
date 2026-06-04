# Qualitative Failure Analysis

상태: **예정 (Phase 5, 논문 draft 단계)**

## 목적

정량 수치만으로는 드러나지 않는 failure mode 를 panel 로 제시. Reviewer
설득력 + 일반화 주장 보강.

## Panel 설계

```
카테고리           예시                              목적
────────────────────────────────────────────────────────────────────
근거리 정면         카메라 1 m 전방                  easy baseline
원거리              3 m 이상                         scale robustness
가림 (partial)      포크 / 사람 / 적재물 가림        occlusion robustness
저조도              실내 어두움                      lighting robustness
unseen 종류         학습 안 한 플라스틱 팔레트         일반화
목재                plastic vs wood                   domain 외삽
```

각 panel 의 column = 모델, row = 카테고리 × 샘플. 추천 구성:

```
                 v8_A (anchor)    ST no filter     ST RANSAC (Ours)
카테고리 1 (2장)  [img][img]       [img][img]       [img][img]
카테고리 2 (2장)  [img][img]       [img][img]       [img][img]
...
카테고리 6 (2장)  [img][img]       [img][img]       [img][img]

총 3 × 12 = 36 overlay 이미지
```

## 기대 관찰

- **anchor vs Ours**: 근거리 / 정면은 둘 다 OK, 가림 / 원거리 / unseen 에서
  Ours 가 개선. Anchor 는 keypoint 부분 누락, Ours 는 완전 검출
- **ST no filter vs Ours**: no filter 는 noisy PL 로 학습해 오히려 특정
  카테고리에서 악화. Ours 는 깨끗한 PL 덕에 일관됨
- **실패 케이스 (Ours 도 틀림)**: 극심한 occlusion, 카메라 극각 → future
  work 섹션에서 언급

## 생성 스크립트

`scripts/data_prep/visualize_inference.py --panel-mode`:

```bash
python scripts/data_prep/visualize_inference.py \
    --weights weights/v8_A_control/final_net_epoch_0068.pth \
              output/st_rounds_none/round_01.pth \
              output/st_rounds_ransac/round_01.pth \
    --test_dir data/pallet/real_data/real_test_seen \
    --panel_categories close,far,occlusion,dark,unseen,wood \
    --num_per_cat 2 \
    --out_dir _docs/experiments/figures/qualitative_panel
```

> `--panel-mode` 는 현재 `visualize_inference.py` 에 없음 — 스크립트 보강
> 필요 (Phase 5 진입 시).

## 선행 조건

- Real Seen / Unseen / Wood 촬영 완료 ([`seen_unseen.md`](./seen_unseen.md))
- Filter ablation 학습 결과 3 가지 ([`../filter/ablation.md`](../filter/ablation.md))
- `visualize_inference.py --panel-mode` 구현

## 관련

- 시각화 스크립트: `scripts/data_prep/visualize_inference.py`
- Real 촬영: `data/pallet/real_data/README.md`
