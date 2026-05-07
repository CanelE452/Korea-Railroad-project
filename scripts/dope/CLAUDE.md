# scripts/dope/

DOPE 모델 실시간 추론. RealSense D435i 카메라 연동. **Native 실행** (Docker 불필요).

## 스크립트

| 파일 | 설명 | 사용법 |
|------|------|--------|
| `run_dope_live.py` | RealSense D435i로 실시간 팔레트 6D 포즈 추정 | `python run_dope_live.py --realsense --weights <path>` |

## 키 조작 (run_dope_live.py)

- `q` — 종료
- `s` — 현재 프레임 저장
- `b` — belief map 토글
- `r` — 결과 오버레이 토글
- belief map 클릭 → 해당 keypoint 상세 정보

## 실행 환경

- conda env: `pallet-pose` (`conda activate pallet-pose`)
- 추가 의존성: `pip install pyrealsense2` + Intel RealSense SDK (Windows installer)
- 카메라: USB 직결 (D435i)
- 가중치: HF Hub 에서 받거나 `config/default.yaml`의 `train.finetune.output_dir` 참조
