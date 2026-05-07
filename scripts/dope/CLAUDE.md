# scripts/dope/

DOPE 모델 실시간 추론. RealSense D435i 카메라 연동.

## 스크립트

| 파일 | 설명 | 사용법 |
|------|------|--------|
| `run_dope_live.py` | RealSense D435i로 실시간 팔레트 6D 포즈 추정 | Docker 컨테이너 내 실행 (`docker compose up`) |

## 키 조작 (run_dope_live.py)

- `q` — 종료
- `s` — 현재 프레임 저장
- `b` — belief map 토글
- `r` — 결과 오버레이 토글
- belief map 클릭 → 해당 keypoint 상세 정보

## 실행 환경

- Docker: `docker-compose.yml` (호스트의 RealSense USB 연결 필요)
- 원클릭 런처: `scripts/launch_v2.ps1` (usbipd USB attach → Docker build → compose up)
- 가중치: `config/default.yaml`의 `train.pretrain.output_dir` 또는 `train.finetune.output_dir` 참조
