#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/kitti_style_dataset [output_dir]"
  echo "       $0 ./datasets/inference_v4"
  echo
  echo "Environment overrides:"
  echo "  SPLIT=test|val|trainval          ImageSets split name. Default: test"
  echo "  DATASET_SUBDIR=testing|training Dataset subdir under the root. Default: testing"
  echo "  DEVICE=cuda|cpu                 Inference device. Default: cuda"
  echo "  FP16=1|0                        Use CUDA autocast FP16. Default: 0"
  echo "  RUN_KITTI_EVAL=1                Also run KITTI metric evaluation. Default: predictions only"
  echo "  ZIP_OUTPUT=1|0                  Zip prediction txt files. Default: 1"
  exit 2
fi

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "$1" && pwd)"
OUTPUT_ARG="${2:-"$BUNDLE_DIR/outputs/inference_$(date +%Y%m%d_%H%M%S)"}"
mkdir -p "$OUTPUT_ARG"
OUTPUT_DIR="$(cd "$OUTPUT_ARG" && pwd)"

SPLIT="${SPLIT:-test}"
DATASET_SUBDIR="${DATASET_SUBDIR:-testing}"
DEVICE="${DEVICE:-cuda}"
FP16="${FP16:-0}"
ZIP_OUTPUT="${ZIP_OUTPUT:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-"$BUNDLE_DIR/weights/model_final.pth"}"
CONFIG_FILE="configs/smoke_geometry_v2_real_v4_distill_dla18_nodcn_640x480.yaml"
SMOKE_DIR="$BUNDLE_DIR/SMOKE-master"
RUNTIME_KITTI="$OUTPUT_DIR/_runtime_kitti"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

if [[ -d "$DATASET_ROOT/image_2" ]]; then
  SOURCE_SPLIT_DIR="$DATASET_ROOT"
else
  SOURCE_SPLIT_DIR="$DATASET_ROOT/$DATASET_SUBDIR"
fi

for required_dir in image_2 calib ImageSets; do
  if [[ ! -d "$SOURCE_SPLIT_DIR/$required_dir" ]]; then
    echo "Missing $required_dir under $SOURCE_SPLIT_DIR" >&2
    exit 1
  fi
done

if [[ ! -f "$SOURCE_SPLIT_DIR/ImageSets/$SPLIT.txt" ]]; then
  echo "Missing split file: $SOURCE_SPLIT_DIR/ImageSets/$SPLIT.txt" >&2
  exit 1
fi

mkdir -p "$RUNTIME_KITTI" "$SMOKE_DIR/datasets"

if [[ -e "$RUNTIME_KITTI/testing" && ! -L "$RUNTIME_KITTI/testing" ]]; then
  echo "Refusing to replace non-symlink: $RUNTIME_KITTI/testing" >&2
  exit 1
fi
ln -sfn "$SOURCE_SPLIT_DIR" "$RUNTIME_KITTI/testing"

if [[ -e "$SMOKE_DIR/datasets/kitti" && ! -L "$SMOKE_DIR/datasets/kitti" ]]; then
  echo "Refusing to replace non-symlink: $SMOKE_DIR/datasets/kitti" >&2
  exit 1
fi
ln -sfn "$RUNTIME_KITTI" "$SMOKE_DIR/datasets/kitti"

if [[ "${RUN_KITTI_EVAL:-0}" == "1" ]]; then
  export SMOKE_SKIP_KITTI_EVAL=0
else
  export SMOKE_SKIP_KITTI_EVAL=1
fi

if [[ "$FP16" == "1" ]]; then
  TEST_FP16=True
else
  TEST_FP16=False
fi

export PYTHONPATH="$SMOKE_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$SMOKE_DIR"
"$PYTHON_BIN" tools/plain_train_net.py \
  --eval-only \
  --config-file "$CONFIG_FILE" \
  --ckpt "$CKPT" \
  DATASETS.TEST_SPLIT "$SPLIT" \
  DATASETS.DETECT_CLASSES "('Car',)" \
  MODEL.DEVICE "$DEVICE" \
  TEST.FP16 "$TEST_FP16" \
  OUTPUT_DIR "$OUTPUT_DIR"

PRED_DIR="$OUTPUT_DIR/inference/kitti_test/data"
echo
echo "Prediction files:"
echo "$PRED_DIR"

if [[ "$ZIP_OUTPUT" == "1" ]]; then
  ZIP_PATH="$OUTPUT_DIR/inference/kitti_test/${SPLIT}_predictions_student_dla18_nodcn_${TEST_FP16}.zip"
  cd "$OUTPUT_DIR/inference/kitti_test"
  if command -v zip >/dev/null 2>&1; then
    zip -qr "$ZIP_PATH" data
  else
    "$PYTHON_BIN" -m zipfile -c "$ZIP_PATH" data
  fi
  echo "Prediction zip:"
  echo "$ZIP_PATH"
fi
