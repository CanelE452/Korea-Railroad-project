# diag-PASS PL contact sheet (도메인별 통과 수도레이블 시각화)

## 목적
diag 필터가 **통과시킨 PL** 만 도메인별로 모아 contact sheet. "필터가 뭘 통과시켰나" 확인용.

## 산출물/스크립트
- 스크립트: `scripts/data_prep/eval/diag_pass_contact_sheet.py` (추론 X, `_full_s2.json` 재활용).
- 출력: `data/pallet/eval_results/filter_domain_analysis/diag_pass_overlays/{outside,night,indoor}_diag_pass.png`
- `_full_{tag}.json` 레코드: frame,img,n_detected,mean_match_px(reproj),kp(9x2 or None),gt8(8x2),filters{fullkp,diag,ratio,ransac_loo,combo},good.
  → kp/gt8 들어있어서 overlay 재생성에 추론 불필요. mean_match=order-free Hungarian reproj.

## 결과 (diag pass, good=reproj<10px)
- outside: pass 27, good 15, median 9.91px (제외 1프레임은 diag fail이라 영향 없음).
- night: pass 10, good 6, median 8.27px. 하위 3장(12~16px) skew 통과.
- indoor: pass 7, good 0, median 12.22px. **전부 통과인데 정밀도 낮음** — PL cuboid가 팔레트보다 약간 큰 scale-skew 일관 발생. diag는 대각선 교차만 보니 등방 scale-skew는 못 거름.

## 시각화 규칙 (사용자 선호)
- PL=메인: cuboid wireframe 밝게(2px) + 9 keypoint 인덱스별 색 + 번호.
- GT=비교 참고: 옅은 녹색 1px.
- best(reproj 낮은 것) 먼저 정렬, 통과한 skew 케이스도 그대로 섞어 표시(품질 분포 보여줘야).
- 셀 상단 banner: reproj/good/ndet. 시트 상단 strip: 도메인 N/good/median + 범례.

## 제외 처리
- `data/_eval_sets/_exclude.txt` 생성. 1778652125245035520 (outside, user-confirmed bad manual GT 2026-06-04). 향후 평가 자동제외용. 이 프레임은 diag도 fail(reproj 20.99)이라 시트엔 원래 미포함.
