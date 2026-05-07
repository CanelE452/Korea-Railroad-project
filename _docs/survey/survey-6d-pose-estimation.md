# 6D Pose Estimation Field Survey

> 6D object pose estimation 분야의 주요 논문/프로젝트 접근법을 비교 정리한다.
> 새로운 실험 설계나 구현 결정 시 참고 자료로 활용한다.
> 생성일: 2026-03-21

## 현재 프로젝트 설정

```
항목          설정
────────────────────────────────────────────────────────────────────────────────
방법론        DOPE (keypoint-based) + Geometric Self-Training
Backbone      VGG-19 (ImageNet pretrained)
출력          9 belief maps + 16 affinity fields, 50×50
Loss          MSE (belief + affinity, 6-stage intermediate supervision)
Sigma         4.0 (belief map Gaussian std)
PnP           EPnP + RANSAC
학습          Synthetic pretrain → Geometric filter pseudo-label → Mixed finetuning
합성 데이터   Isaac Sim 4.5 + Replicator, 도메인 랜덤화
평가          PCK@3px (val), ADD / 5cm5° / Reproj (test)
```

---

## 1. Pose Representation

6D 포즈를 어떻게 표현하고 추출하느냐에 따라 크게 4가지 패러다임으로 나뉜다.

```
패러다임                대표 방법                    출력                          포즈 복원          장점                              단점
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Keypoint-based          DOPE, PVNet                  2D keypoint heatmap/voting    PnP solver         해석 가능, 멀티인스턴스, 가볍다   keypoint 수에 의존, 대칭 객체 어려움
Dense correspondence    CDPN, ZebraPose, GDR-Net     픽셀별 3D 좌표 (NOCS map)     PnP/RANSAC         폐색에 강함, 정밀도 높음          계산량 큼, 대칭 ambiguity
Direct regression       PoseCNN                      쿼터니언 + 평행이동           없음 (end-to-end)  단순, 빠름                        정밀도 낮음, 비선형 회전 공간 학습 어려움
Render-and-compare      FoundationPose, MegaPose     SE(3) delta                   반복 정제          새 객체 일반화, CAD만 있으면 됨   느림, 렌더링 필요
```

### 현재 프로젝트와 비교
- **현재**: Keypoint-based (DOPE) — 팔레트는 비대칭 직육면체로 keypoint 방식에 적합
- **대안**: Dense correspondence (GDR-Net)는 정밀도 더 높지만 학습 복잡도 증가
- **Render-and-compare** (FoundationPose)는 CAD 모델 있으면 zero-shot 가능하나 실시간성 부족

---

## 2. 학습 전략 (Supervised vs Self-Training vs DA)

```
전략                          대표 방법                    Real 라벨 필요     핵심 메커니즘                       성능 수준
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Supervised                    PVNet, GDR-Net, ZebraPose    전량               GT 포즈 직접 학습                   최고 (상한선)
Synthetic-only                DOPE (original)              없음               도메인 랜덤화로 sim-to-real gap 극복 하한선
Differentiable rendering      Self6D, Self6D++             없음 (RGB-D 필요)  렌더링 일관성 loss로 자기지도       Supervised에 근접
Pseudo-label self-training    DSC-PoseNet, Ours            없음               모델 예측 → 필터링 → 재학습         Syn-only 대비 큰 향상
Foundation model              FoundationPose, MegaPose     없음               대규모 사전학습 + 일반화            객체별 학습 불필요
```

### 주요 Self-Training 방법 비교

```
방법          Pseudo-label 생성            필터링                                      필요 센서   특징
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Self6D        Differentiable rendering     렌더링 loss 수렴                            RGB-D      깊이 정보 필수
Self6D++      Noisy student + rendering    Teacher-student 일관성                      RGB-D      폐색 인식 추가
DSC-PoseNet   Dual-scale mask 비교         Scale consistency                           RGB        약지도 (bbox만 필요)
Ours          DOPE + EPnP                  3단계 기하학적 필터 (Flip/Diagonal/LOO-PnP) RGB        도메인 지식 활용, 깊이 불필요
```

### 현재 프로젝트와 비교
- **현재**: RGB-only pseudo-label self-training + geometric filter
- **차별점**: Self6D/Self6D++는 RGB-D 필요, 우리는 RGB-only
- **차별점**: DSC-PoseNet은 mask 기반, 우리는 keypoint + 물리적 규격 기반 필터링
- **고려사항**: Self6D++의 noisy student 프레임워크를 우리 파이프라인에 결합 가능성

---

## 3. 합성 데이터 & Domain Randomization

```
항목          Unstructured DR (Tremblay 2018)   Structured DR (Prakash 2019)   현재 프로젝트
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
배경          랜덤 텍스처/이미지                맥락에 맞는 장면               3모드 혼합 (창고 40%, 실내 30%, 야외 30%)
객체 배치     균일 랜덤                         맥락 인식 (차는 도로 위)       바닥 수평 고정, yaw만 자유
조명          랜덤 HDR                          물리 기반                      DomeLight + RectLight 3개, 랜덤화
디스트랙터    랜덤 형상                         의미론적으로 적절              54종 6카테고리 (primitive + 다양화)
가림          랜덤                              맥락에 맞는 적재물             60% 프레임에 팔레트 위 적재물
데이터량      매우 많이 필요                    중간                           5,000~15,000장
```

### 주요 합성 데이터 도구

```
도구                            사용 방법                특징
─────────────────────────────────────────────────────────────────────────────────────
NVIDIA Isaac Sim + Replicator   DOPE, 현재 프로젝트     PathTracing, USD 기반, 프로그래밍 가능
NVISII                          DOPE (이전 버전)        빠른 레이트레이싱, Python API
BlenderProc                     BOP Challenge 데이터 생성 Blender 기반, 유연함
Kubric                          Google, 다양한 객체     물리 시뮬레이션 포함
```

### 현재 프로젝트와 비교
- **현재**: Isaac Sim 4.5 + Replicator, Structured DR에 가까움 (도메인 특화 배치)
- **강점**: 카메라를 리프터 시점으로 제한, 팔레트 물리적 배치 반영
- **개선 가능**: LLM-aided 텍스처 다양화 (FoundationPose 방식)

---

## 4. 네트워크 구조

```
방법              Backbone               Head 구조                       출력 해상도   파라미터
─────────────────────────────────────────────────────────────────────────────────────────────────
DOPE              VGG-19 (24층)          6-stage CPM                     50×50         ~60M
PVNet             ResNet-18              Encoder-decoder                 입력 해상도   ~12M
CDPN              ResNet                 Encoder-decoder × 2             64×64         ~25M
GDR-Net           ResNet-34 / ConvNeXt   3 geometric maps + Patch-PnP   64×64         ~35M
ZebraPose         ResNet / EfficientNet  FCN encoder-decoder             64×64         ~25M
FoundationPose    Transformer            Refiner + Scorer                patch-based   ~100M+
```

### Backbone 선택 가이드

```
Backbone        특징                       적합한 경우
──────────────────────────────────────────────────────────────────────────
VGG-19          큰 receptive field, 느림   multi-stage refinement (DOPE)
ResNet-18/34    가볍고 효율적              실시간 필요 시
ResNet-50/101   더 높은 표현력             정밀도 우선
ConvNeXt        최신 CNN, ResNet 대체      GDRNPP에서 검증
EfficientNet    효율-성능 최적화           엣지 디바이스 배포
Transformer     전역 attention             Foundation model, 대규모 학습
```

### 현재 프로젝트와 비교
- **현재**: VGG-19 + 6-stage CPM (DOPE 원본 그대로)
- **고려사항**: ResNet-34로 교체 시 속도↑ + 성능 유지 가능 (GDR-Net 참고)

---

## 5. Loss Function & Belief Map 설계

### Loss Function 비교

```
방법              Loss                                      특징
──────────────────────────────────────────────────────────────────────────────────────
DOPE              MSE (belief + affinity)                   단순, 6-stage 중간 감독
PVNet             Smooth L1 (voting) + CE (seg)             견고한 voting
CDPN              L1 (coord) + regression (trans)           회전/평행이동 분리
GDR-Net           L1 + CE + PM loss (6D rotation)           기하학적 가이드
ZebraPose         BCE (hierarchical bits) + L1 (mask)       Coarse-to-fine 가중치
Self6D            MSE + rendering losses (RGB/depth/mask)   미분가능 렌더링
FoundationPose    Contrastive triplet + SE(3) regression    포즈 스코어링
```

### Belief Map Sigma 설정

```
설정     Sigma      커버리지 (50×50 기준)   효과
────────────────────────────────────────────────────────────────────────────────
극소     0.5        ~1px (0.04%)            Gradient vanishing, 학습 실패
소       2.0        ~13×13 (7%)             학습 가능, 정밀도 높음
표준     4.0        ~25×25 (25%)            DOPE 공식 기본값, 안정적 학습
대       7.0        ~43×43 (74%)            OpenPose 기본값, 쉬운 학습 but 낮은 정밀도
적응형   거리 비례  가변                    최신 연구, 스케일 변화에 강함
```

> **일반 원칙**: sigma↑ = 학습 용이 + 정밀도↓, sigma↓ = 학습 어려움 + 정밀도↑

### 현재 프로젝트와 비교
- **현재**: MSE loss + sigma=4.0 (표준)
- **self-training 단계**: sigma=2.0 (config), pretrain보다 작은 sigma로 정밀도 높임
- **고려사항**: 적응형 sigma (거리 기반)는 팔레트 크기 변화가 클 때 유용

---

## 6. PnP Solver & 후처리

```
Solver           필요 점 수   복잡도   미분 가능   사용처
──────────────────────────────────────────────────────────────────────────────
P3P              3            O(1)     No          RANSAC 내부
EPnP             ≥4           O(n)     No          DOPE, PVNet, 현재 프로젝트
DLT              ≥6           O(n)     No          고전적 방법
Iterative (LM)   ≥4           반복     No          ICP 정제
BPnP             N            반복     Yes         End-to-end 학습
EPro-PnP         N            반복     Yes         CVPR 2022 Best Student Paper
Progressive-X    N            반복     No          ZebraPose, 멀티인스턴스
```

### 후처리 방법

```
방법                            입력              효과                        비용
────────────────────────────────────────────────────────────────────────────────────────
ICP (Iterative Closest Point)   RGB-D + CAD       PoseCNN에서 +17% ADD-S      느림, 깊이 필요
RANSAC outlier 제거             2D keypoints      잘못된 keypoint 필터링      빠름
Pose hypothesis 랭킹            복수 후보         FoundationPose 스코어러     중간
Geometric filter                2D/3D keypoints   현재 프로젝트 핵심 기여     빠름
```

### 현재 프로젝트와 비교
- **현재**: EPnP + RANSAC (threshold=8px, 100 iter) + 3단계 geometric filter
- **차별점**: 대부분의 방법은 confidence threshold만 사용, 우리는 기하학적 일관성 검증 추가
- **고려사항**: EPro-PnP 적용 시 end-to-end 학습 가능 (연구 확장)

---

## 7. 평가 메트릭

### 메트릭 정의

```
메트릭          정의                                     임계값             사용처
──────────────────────────────────────────────────────────────────────────────────────────
ADD             모델 점들의 평균 3D 거리                 <0.1d (직경 10%)   비대칭 객체
ADD-S           최근접 점 매칭 평균 거리                 <0.1d              대칭 객체
ADD(-S) AUC     ADD/ADD-S 커브 아래 면적                 —                  YCB-Video 표준
5cm-5°          평행이동 <5cm AND 회전 <5°               5cm, 5°            로봇 조작
Reproj          2D 재투영 오차                           <5px               빠른 평가
PCK@Npx         N px 이내 keypoint 비율                  3px, 5px, 10px     Keypoint 정확도
VSD             Visible Surface Discrepancy              τ, δ               BOP Challenge
MSSD            Max Symmetry-aware Surface Distance      —                  BOP 2022+
MSPD            Max Symmetry-aware Projection Distance   —                  BOP 2022+
```

### BOP Challenge 표준 (AR = Average Recall)
```
AR = (AR_VSD + AR_MSSD + AR_MSPD) / 3
```
7개 코어 데이터셋: LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V

### 현재 프로젝트와 비교
- **현재**: PCK@3px (val), ADD + 5cm5° + Reproj (test)
- **BOP 표준과 차이**: BOP는 AR(VSD+MSSD+MSPD) 사용
- **팔레트 특성**: 비대칭 → ADD 적합, 대형 → 5cm5°가 실용적 지표

---

## 8. Pseudo-Label 필터링 전략

Self-training의 핵심은 pseudo-label 품질. 필터링 전략 비교:

```
방법                        필터링 기준                  센서    도메인 지식
─────────────────────────────────────────────────────────────────────────────────
Confidence threshold        모델 출력 confidence         RGB     없음
Self6D++ (noisy student)    Teacher-student 일관성       RGB-D   없음
DSC-PoseNet                 Dual-scale pose 일관성       RGB     없음
Ours (A)                    Flip consistency             RGB     없음
Ours (B)                    Diagonal concurrency         RGB     직육면체 가정
Ours (C)                    Leave-one-out PnP stability  RGB     PnP 기반 pose 검증
```

### 필터링 품질 vs 채택률 트레이드오프

```
엄격한 필터 → 높은 PL 정확도 / 낮은 채택률 → 학습 데이터 부족 위험
느슨한 필터 → 낮은 PL 정확도 / 높은 채택률 → 노이즈 라벨로 성능 저하
```

- **Confidence-only**: 가장 단순하지만 overconfident 예측 필터링 못함
- **기하학적 필터 (Ours)**: 모델 confidence와 독립적으로 물리적 일관성 검증 → 상호 보완
- **Self6D++ 방식**: Teacher가 좋아야 student도 좋음 → 초기 품질에 민감

### 현재 프로젝트와 비교
- **현재**: 3단계 기하학적 필터 (A: flip consistency, B: diagonal concurrency, C: LOO PnP)
- **강점**: RGB-only, 도메인 지식 활용, confidence와 독립적
- **약점**: 객체 형태에 의존 (직육면체 가정), 다른 객체에는 C 필터 재설계 필요

---

## Summary: 현재 프로젝트 vs 일반적 접근

```
항목            현재 프로젝트              일반적 접근                   비고
──────────────────────────────────────────────────────────────────────────────────────────────
Pose 표현       Keypoint (DOPE)            Dense correspondence가 주류  팔레트에는 keypoint 충분
Backbone        VGG-19                     ResNet-34/50이 주류          VGG는 무거움, 교체 고려
Loss            MSE                        Task-specific loss 조합      MSE로 충분 (keypoint 방식)
Sigma           4.0                        2~7 범위                     표준 설정
PnP             EPnP + RANSAC              EPnP + RANSAC (동일)         표준
Self-training   Geometric filter           Rendering loss / Noisy student  RGB-only가 차별점
PL 필터         3단계 기하학적             Confidence threshold         핵심 기여
합성 데이터     Isaac Sim + Structured DR  BlenderProc / Isaac Sim      표준
평가            ADD + 5cm5°                BOP AR 또는 ADD              표준에 가까움
```

---

## 추가 조사 필요 항목

- [ ] ResNet-34 backbone으로 DOPE 교체 시 성능/속도 비교
- [ ] 적응형 sigma (거리 기반) 적용 가능성
- [ ] EPro-PnP (differentiable PnP) 적용으로 end-to-end 학습 가능성
- [ ] Self6D++ noisy student + 우리 geometric filter 결합 실험
- [ ] BOP 표준 메트릭 (AR) 추가 구현 및 벤치마크 비교
- [ ] FoundationPose의 LLM-aided 텍스처 다양화를 합성 데이터에 적용

---

## 참고 문헌

```
약칭              논문                                                                                  학회
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DOPE              Tremblay et al., "Deep Object Pose Estimation"                                        CoRL 2018
PoseCNN           Xiang et al., "PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation" RSS 2018
PVNet             Peng et al., "PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation"              CVPR 2019
CDPN              Li et al., "CDPN: Coordinates-based Disentangled Pose Network"                        ICCV 2019
Self6D            Wang et al., "Self6D: Self-Supervised Monocular 6D Object Pose Estimation"            ECCV 2020
GDR-Net           Wang et al., "GDR-Net: Geometry-Guided Direct Regression Network"                     CVPR 2021
DSC-PoseNet       Yang et al., "DSC-PoseNet: Learning 6DoF Object Pose via Dual-Scale Consistency"      CVPR 2021
Self6D++          Wang et al., "Occlusion-Aware Self-Supervised Monocular 6D Object Pose Estimation"    TPAMI 2022
ZebraPose         Su et al., "ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation" CVPR 2022
EPro-PnP          Chen et al., "EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points"    CVPR 2022
MegaPose          Labbé et al., "MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare"    CoRL 2022
FoundationPose    Wen et al., "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects" CVPR 2024
BOP 2023          Sundermeyer et al., "BOP Challenge 2023 on Detection, Segmentation and Pose Estimation" CVPRW 2024
```
