# challenge/model — DOPE 가중치 보관 폴더

이 폴더는 `.gitignore` 로 가중치 (*.pth) 만 제외되고 폴더 자체는 유지됩니다.
git pull 받은 후 아래 가중치를 직접 복사/다운로드해 두세요.

## 필요 파일

### `challengenight.pth` (필수)

- **용도**: `25y_automatic_lifter-master/.../depth_cam/main_rec.py` 의 DOPE 6D pose 추론
- **크기**: ~192 MB
- **참조 코드**: `calib/config.py` 의 `MODEL_PATH_6D`
- **학습 데이터**: synthetic (Blender + Isaac Sim) + capture night 데이터로 fine-tune
- **모델 구조**: DopeNetwork (VGG-19 backbone, 9 belief + 16 affinity)

## 다운로드 방법

### 옵션 1 — HuggingFace (권장)

```bash
# (upload 스크립트는 depth_cam/tools/hf_upload_challengenight.py 참조)
huggingface-cli download <repo>/<model> challengenight.pth \
    --local-dir challenge/model
```

### 옵션 2 — USB / 파일 공유

운영 PC 또는 학습 PC 에서 직접 복사.

### 옵션 3 — 별도 학습

`scripts/train_dope.sh` + `challenge/scripts/finetune.sh` 로 새로 학습.
`challenge/config/task.yaml` 의 `finetune` 섹션 참조.

## 사용 확인

```bash
ls -la challenge/model/challengenight.pth   # 파일 존재 + 약 192 MB
cd 25y_automatic_lifter-master/25y_automatic_lifter-master/depth_cam
conda activate pallet-pose
python main_rec.py
# 로그에 "✅ DOPE 6D pose estimator 초기화 완료" 가 나오면 OK
```

`FileNotFoundError: DOPE weights not found` 에러가 나오면 위 파일 위치/이름 확인.
