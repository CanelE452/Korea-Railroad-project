# Phase 1 — 3차 발표용 Self-Training 반복 + 도메인 갭 실험

상태: **진행중** (2026-05-27 ~ 2026-05-29)

## 배경

2차 발표 한계점 (본인 인정):
1. 학습/평가가 동일 실내 환경 → 도메인 갭 미검증
2. Self-training 반복 (R1→R2→R3) 미실험 (F5 는 R1 단발)

3차 발표는 강의 공지상 "구현" 단계. 위 두 한계를 직접 보완하는 실험 트랙을 설계.

challenge/ 폴더 작업물은 **사용 안 함** (과제용 과적합, 논문 외 영역).

## 핵심 메시지

```
1. Self-training 반복 (R0→R1→R2) 으로 성능 단조 증가/수렴
2. 학습 도메인(실내)과 다른 환경(실외/야간)에서도 robust
```

## 실험 매트릭스

```
              R0 (anchor)       R1 (real-only PL)         R2 (real-only PL 교체)
─────────────────────────────────────────────────────────────────────────────────────
indoor        ep65 (기존)       F5 (기존, NN<20px 60.5%)  NEW: R1_indoor 로 PL 재추출 + ft
outside       ep65              NEW: outside PL ft          NEW: R1_outside 로 PL 재추출 + ft
night         ep65              NEW: night PL ft            NEW: R1_night 로 PL 재추출 + ft
```

- 모든 트랙 anchor = `weights/v8_ablation_A_coord/final_net_epoch_0065.pth`
- 모든 ft = real-only, 96 ep, lr=5e-5, batch=4, sigma=4.0 (F5 패턴)
- indoor R1 = F5 재사용 (학습 1회 절약)
- 새 학습 = 5회 (outside R1/R2, night R1/R2, indoor R2)

## 데이터 매핑

```
트랙       Unlabeled PL pool                        평가 GT
───────────────────────────────────────────────────────────────────────────────
indoor     data/pallet/raw_data/real_pool_all/      F5 평가셋 (capture0403middle + gt_final_isaac)
outside    data/outside/capturepallet01~11+cad/rgb  challenge/data/cp01~08_manual_gt
night      data/night/capturenight01~10/rgb         challenge/data/cn01,03~09_manual_gt
```

## 필터 — Fallback 정책

```
Plan A   RANSAC subset consensus (n=50, k=5, τ=5px, c≥6) — 기본
Plan C   Plan A 통과 0장이면 F5 모델로 outside/night 추론 (시작점 변경)
```

## 메트릭

```
Primary     NN matching <20px  (F5 라인과 비교 가능)
Secondary   PCK@5/10px,  PnP reprojection error mean
```

## 일정

```
Day 0   데이터 형식 검증 + R0 추론/필터/PL 추출 + R0 baseline 3 도메인 평가
Day 1   R1 학습 (outside, night) + R1 평가 + R2 PL 재추출
Day 2   R2 학습 (3 트랙) + R2 평가 + 라운드별 곡선 figure
Day 3   정성 시각화 + 프로토타입 데모 + 슬라이드 정리 (Phase 2)
```

## 체크포인트

```
#1 (Day 0 끝)   outside/night PL ≥ 1장 ✓, baseline 3 셀 채움
#2 (Day 1 끝)   R1 대각선 셀 > R0 대각선 셀
#3 (Day 2 끝)   R2 대각선 셀 ≥ R1 대각선 셀 (수렴 또는 향상)
```

## Phase 2 (보조 작업, Phase 1 완료 후)

```
A. 평가 셀 보강     R0 baseline (필수, Phase 1 에서 함) + cross-domain transfer (선택)
B. 정성 시각화     성공/실패 panel, ST 적용 전후 keypoint overlay
C. 프로토타입 데모  이미지 → keypoint → PnP → 6D pose 1 cycle
D. 슬라이드 정리   loss ablation, 필터 selection, ST 곡선
E. 부가 figure    PL 수/채택률 곡선, 시퀀스 샘플
```

## 관련 문서

- F5 (R1 단발) 결론: `_docs/experiments/self_training/rounds.md`
- 필터 선정: `_docs/filter/2026-04-11_selection.md`
- Loss ablation: `_docs/models/v8_ablation.md`
- 모델 카탈로그: `_docs/experiments/model_catalog.md`
