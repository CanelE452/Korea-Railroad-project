#!/bin/bash
# challenge/scripts/finetune.sh
# challenge/data/{train,val}을 사용해 baseline에서 추가 ft.
# 실제 학습은 메인 scripts/train_dope.sh에 위임 (논리는 한 곳에서만 유지).
#
# 사용법:
#   bash challenge/scripts/finetune.sh
#   bash challenge/scripts/finetune.sh --exp_name challenge_ft_v2
#
# 데이터 준비 후에 호출. challenge/data/train/ 이 비어있으면 안내 후 종료.

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_DIR="challenge/data/train"
VAL_DIR="challenge/data/val"
NET_PATH="challenge/weights/baseline_v8_A.pth"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A "$TRAIN_DIR" 2>/dev/null)" ]; then
    echo "[STUB] $TRAIN_DIR 에 학습 데이터(NDDS 포맷, {i:06d}.png + {i:06d}.json)가 없습니다."
    echo "       challenge 데이터 수집 후 다시 실행하세요."
    echo "       참고: challenge/README.md, data/pallet/real_data/README.md"
    exit 0
fi

if [ ! -f "$NET_PATH" ]; then
    echo "[ERROR] baseline weight 없음: $NET_PATH"
    exit 1
fi

EXP_NAME="${EXP_NAME:-challenge_ft}"

bash scripts/train_dope.sh --finetune \
    --net_path "$NET_PATH" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --exp_name "$EXP_NAME" \
    --symmetric_loss \
    --struct_loss --struct_coord 0.003 \
    "$@"

# 산출물을 challenge/weights/finetuned/ 로 링크 (없으면 복사)
SRC="weights/$EXP_NAME"
DST="challenge/weights/finetuned"
mkdir -p "$DST"
if [ -d "$SRC" ]; then
    cp -u "$SRC"/*.pth "$DST"/ 2>/dev/null || true
    echo "[DONE] Weights copied → $DST/"
fi
