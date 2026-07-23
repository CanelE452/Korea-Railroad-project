#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$BUNDLE_DIR/datasets/inference_v4"
OUTPUT_DIR="${1:-"$BUNDLE_DIR/outputs/inference_v4_student_fp32_$(date +%Y%m%d_%H%M%S)"}"

export DATASET_SUBDIR="${DATASET_SUBDIR:-testing}"
export SPLIT="${SPLIT:-test}"
export FP16="${FP16:-0}"

"$BUNDLE_DIR/run_inference.sh" "$DATASET_ROOT" "$OUTPUT_DIR"
