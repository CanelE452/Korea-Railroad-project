#!/bin/bash
# DOPE 학습 실행 — config/default.yaml 기반
#
# 사용법:
#   bash scripts/train_dope.sh                    # scratch (pretrain)
#   bash scripts/train_dope.sh --finetune         # fine-tune from latest weight
#   bash scripts/train_dope.sh --finetune --net_path weights/custom/net.pth
#   bash scripts/train_dope.sh --exp_name blender_v1 --train_dir data/pallet/training_data/blender_train_v1
#   bash scripts/train_dope.sh --exp_name blender_ft --finetune --train_dir data/pallet/training_data/blender_train_v1
#
# 환경: conda activate pallet-pose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$PROJECT_ROOT/config/default.yaml"

# --- YAML 파서 (python one-liner) ---
yq() {
    python -c "
import yaml, sys, os
cfg_path = os.path.normpath(sys.argv[2])
with open(cfg_path, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
keys = sys.argv[1].split('.')
v = cfg
for k in keys:
    v = v[k]
print(v)
" "$1" "$CONFIG"
}

export PYTHONUNBUFFERED=1
PYTHON_EXE="${PYTHON_EXE:-$(yq paths.python_exe)}"

# --- 인자 파싱 ---
FINETUNE=false
NET_PATH=""
EXP_NAME="${EXP_NAME:-}"
TRAIN_DIR_OVERRIDE=""
TRAIN_DIRS_OVERRIDE=""
VAL_DIR_OVERRIDE=""
GEO_LOSS=false
GEO_LAMBDA=""
GEO_WARMUP=""
VIS_COORD_LOSS=false
VIS_LAMBDA=""
VIS_WARMUP=""
REL_LOSS=false
REL_LAMBDA=""
REL_WARMUP=""
STRUCT_LOSS=false
STRUCT_LAMBDA=""
STRUCT_WARMUP=""
STRUCT_FLIP=""
STRUCT_EDGE=""
STRUCT_COORD=""
SYMMETRIC_LOSS=false
TRUNCATION_AUG_PROB="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --finetune) FINETUNE=true; shift ;;
        --net_path) NET_PATH="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --train_dir) TRAIN_DIR_OVERRIDE="$2"; shift 2 ;;
        --train_dirs) TRAIN_DIRS_OVERRIDE="$2"; shift 2 ;;
        --val_dir) VAL_DIR_OVERRIDE="$2"; shift 2 ;;
        --geo_loss) GEO_LOSS=true; shift ;;
        --geo_lambda) GEO_LAMBDA="$2"; shift 2 ;;
        --geo_warmup) GEO_WARMUP="$2"; shift 2 ;;
        --vis_coord_loss) VIS_COORD_LOSS=true; shift ;;
        --vis_lambda) VIS_LAMBDA="$2"; shift 2 ;;
        --vis_warmup) VIS_WARMUP="$2"; shift 2 ;;
        --rel_loss) REL_LOSS=true; shift ;;
        --rel_lambda) REL_LAMBDA="$2"; shift 2 ;;
        --rel_warmup) REL_WARMUP="$2"; shift 2 ;;
        --struct_loss) STRUCT_LOSS=true; shift ;;
        --struct_lambda) STRUCT_LAMBDA="$2"; shift 2 ;;
        --struct_warmup) STRUCT_WARMUP="$2"; shift 2 ;;
        --struct_flip) STRUCT_FLIP="$2"; shift 2 ;;
        --struct_edge) STRUCT_EDGE="$2"; shift 2 ;;
        --struct_coord) STRUCT_COORD="$2"; shift 2 ;;
        --symmetric_loss) SYMMETRIC_LOSS=true; shift ;;
        --truncation_aug_prob) TRUNCATION_AUG_PROB="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# --- config에서 값 로드 ---
TRAIN_DIR="$(yq data.train_dir)"
OBJECT="$(yq data.object)"
IMAGE_SIZE="$(yq model.input_size)"
SIGMA="$(yq train.sigma)"
WORKERS="${WORKERS:-$(yq train.workers)}"
SAVE_EVERY="$(yq train.save_every)"
LOG_INTERVAL="$(yq train.log_interval)"
MIN_IMAGES="${MIN_IMAGES:-$(yq data.min_train_images)}"

if $FINETUNE; then
    OUTPUT_DIR="$(yq train.finetune.output_dir)"
    EPOCHS="${EPOCHS:-$(yq train.finetune.epochs)}"
    LR="${LR:-$(yq train.finetune.learning_rate)}"
    if [ -z "$NET_PATH" ]; then
        NET_PATH="$(yq train.finetune.pretrained_weights)"
    fi
else
    OUTPUT_DIR="$(yq train.pretrain.output_dir)"
    EPOCHS="${EPOCHS:-$(yq train.pretrain.epochs)}"
    LR="${LR:-$(yq train.pretrain.learning_rate)}"
fi

# --- CLI 오버라이드 적용 ---
if [ -n "$TRAIN_DIR_OVERRIDE" ]; then
    TRAIN_DIR="$TRAIN_DIR_OVERRIDE"
fi
if [ -n "$VAL_DIR_OVERRIDE" ]; then
    VAL_DIR_CLI="$VAL_DIR_OVERRIDE"
fi
if [ -n "$EXP_NAME" ]; then
    OUTPUT_DIR="weights/$EXP_NAME"
fi

# 멀티 디렉토리: --train_dirs "dir1 dir2 dir3" 가 있으면 우선
# 없으면 단일 TRAIN_DIR 만 사용. CleanVisiiDopeLoader 가 path 리스트 + 재귀 탐색 지원.
if [ -n "$TRAIN_DIRS_OVERRIDE" ]; then
    TRAIN_DIRS="$TRAIN_DIRS_OVERRIDE"
else
    TRAIN_DIRS="$TRAIN_DIR"
fi

# 데이터 확인 (멀티 dir 합산, 재귀 탐색)
TRAIN_COUNT=0
for d in $TRAIN_DIRS; do
    n=$(find "$d" -name '*.png' 2>/dev/null | wc -l)
    TRAIN_COUNT=$((TRAIN_COUNT + n))
done
echo "Train images: $TRAIN_COUNT (across $(echo $TRAIN_DIRS | wc -w) dir(s))"
if [ "$TRAIN_COUNT" -lt "$MIN_IMAGES" ]; then
    echo "[ERROR] Too few training images ($TRAIN_COUNT < $MIN_IMAGES). Run generate_all.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd Deep_Object_Pose/train

echo "============================================"
if $FINETUNE; then
    echo " DOPE Fine-tune"
    echo "============================================"
    echo " Pretrain:   ../../$NET_PATH"
else
    echo " DOPE Training (from scratch)"
    echo "============================================"
fi
BATCH_SIZE="${BATCH:-$(if $FINETUNE; then yq train.finetune.batch_size; else yq train.pretrain.batch_size; fi)}"

echo " Config:     ../../$CONFIG"
echo " Data:       $TRAIN_DIRS ($TRAIN_COUNT images)"
echo " Output:     ../../$OUTPUT_DIR"
echo " Object:     $OBJECT"
echo " Epochs:     $EPOCHS"
echo " Batch size: $BATCH_SIZE"
echo " LR:         $LR"
echo " Image size: $IMAGE_SIZE"
echo " Sigma:      $SIGMA"
echo " Workers:    $WORKERS"
echo "============================================"

# --data 부분을 멀티 dir 로 펼침 (각각 ../../ 접두)
DATA_ARGS=""
for d in $TRAIN_DIRS; do
    DATA_ARGS="$DATA_ARGS \"../../$d\""
done

TRAIN_CMD="\"$PYTHON_EXE\" train.py \
    --data $DATA_ARGS \
    --object $OBJECT \
    --epochs $EPOCHS \
    --batchsize $BATCH_SIZE \
    --imagesize $IMAGE_SIZE \
    --lr $LR \
    --save_every $SAVE_EVERY \
    --outf \"../../$OUTPUT_DIR\" \
    --loginterval $LOG_INTERVAL \
    --workers $WORKERS \
    --sigma $SIGMA \
    --truncation_aug_prob $TRUNCATION_AUG_PROB"

if $FINETUNE && [ -f "../../$NET_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --net_path \"../../$NET_PATH\""
    echo "[INFO] Fine-tuning from: $NET_PATH"
elif $FINETUNE; then
    echo "[ERROR] Pretrain weight not found: $NET_PATH"
    exit 1
fi

if $SYMMETRIC_LOSS; then
    TRAIN_CMD="$TRAIN_CMD --symmetric_loss"
    echo "[INFO] Symmetric loss enabled (180° front-back swap)"
fi

if [ "$TRUNCATION_AUG_PROB" != "0" ] && [ "$TRUNCATION_AUG_PROB" != "0.0" ]; then
    echo "[INFO] On-the-fly truncation aug enabled (prob=$TRUNCATION_AUG_PROB)"
fi

if $GEO_LOSS; then
    TRAIN_CMD="$TRAIN_CMD --geo_loss"
    [ -n "$GEO_LAMBDA" ] && TRAIN_CMD="$TRAIN_CMD --geo_lambda $GEO_LAMBDA"
    [ -n "$GEO_WARMUP" ] && TRAIN_CMD="$TRAIN_CMD --geo_warmup $GEO_WARMUP"
    echo "[INFO] Geometric loss enabled"
fi

if $VIS_COORD_LOSS; then
    TRAIN_CMD="$TRAIN_CMD --vis_coord_loss"
    [ -n "$VIS_LAMBDA" ] && TRAIN_CMD="$TRAIN_CMD --vis_lambda $VIS_LAMBDA"
    [ -n "$VIS_WARMUP" ] && TRAIN_CMD="$TRAIN_CMD --vis_warmup $VIS_WARMUP"
    echo "[INFO] Visibility coord loss enabled"
fi

if $REL_LOSS; then
    TRAIN_CMD="$TRAIN_CMD --rel_loss"
    [ -n "$REL_LAMBDA" ] && TRAIN_CMD="$TRAIN_CMD --rel_lambda $REL_LAMBDA"
    [ -n "$REL_WARMUP" ] && TRAIN_CMD="$TRAIN_CMD --rel_warmup $REL_WARMUP"
    echo "[INFO] Reliability loss enabled"
fi

if $STRUCT_LOSS; then
    TRAIN_CMD="$TRAIN_CMD --struct_loss"
    [ -n "$STRUCT_LAMBDA" ] && TRAIN_CMD="$TRAIN_CMD --struct_lambda $STRUCT_LAMBDA"
    [ -n "$STRUCT_WARMUP" ] && TRAIN_CMD="$TRAIN_CMD --struct_warmup $STRUCT_WARMUP"
    [ -n "$STRUCT_FLIP" ] && TRAIN_CMD="$TRAIN_CMD --struct_flip $STRUCT_FLIP"
    [ -n "$STRUCT_EDGE" ] && TRAIN_CMD="$TRAIN_CMD --struct_edge $STRUCT_EDGE"
    [ -n "$STRUCT_COORD" ] && TRAIN_CMD="$TRAIN_CMD --struct_coord $STRUCT_COORD"
    echo "[INFO] Structural loss enabled (flip+edge+coord)"
fi

eval $TRAIN_CMD

echo ""
echo "[DONE] Training complete!"
echo "  Weights: $OUTPUT_DIR/"
echo "  TensorBoard: python scripts/launch_tensorboard.py"

# 자동 Val 평가 (PCK@3/5/10px + PnP Reproj)
cd ../..
VAL_DIR="${VAL_DIR_CLI:-$(yq data.val_dir)}"
EPOCH_TAG=$(printf "%04d" "$EPOCHS")
WEIGHTS="$OUTPUT_DIR/net_epoch_${EPOCH_TAG}.pth"
MAX_FRAMES="$(yq eval.max_frames)"
if [ -d "$VAL_DIR" ] && [ -f "$WEIGHTS" ]; then
    echo ""
    echo "============================================"
    echo " Running Val Evaluation"
    echo "============================================"
    "$PYTHON_EXE" "$(yq paths.eval_script)" \
        --weights "$WEIGHTS" \
        --val_dir "$VAL_DIR" \
        --output_dir "$OUTPUT_DIR/eval_results" \
        --max_frames "$MAX_FRAMES"
fi

# 학습 완료 알림 (ntfy.sh)
NTFY_TOPIC="${NTFY_TOPIC:-$(yq notification.ntfy_topic)}"
if $FINETUNE; then
    MSG="DOPE fine-tune done! (lr=$LR, ${EPOCHS}ep, $TRAIN_COUNT images)"
else
    MSG="DOPE training done! (sigma=$SIGMA, ${EPOCHS}ep, $TRAIN_COUNT images)"
fi
curl -s -H "Content-Type: text/plain; charset=utf-8" -d "$MSG" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
