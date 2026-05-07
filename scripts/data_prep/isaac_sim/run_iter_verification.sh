#!/bin/bash
# Iter 4-10 검증 데이터 일괄 생성 스크립트
# Isaac Sim을 한 번만 실행하는 것이 효율적이지만,
# gen_replicator_data.py는 실행당 하나의 output_dir만 지원하므로
# iteration별로 별도 실행한다.
#
# 사용법:
#   conda activate pallet-pose
#   bash scripts/data_prep/run_iter_verification.sh

set -e

export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/gen_replicator_data.py"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_keypoints.py"
VIS_SCRIPT="$SCRIPT_DIR/visualize_annotations.py"
BASE_DIR="data/pallet"

# Iter 4-10: 각각 다른 시드, 10프레임
SEEDS=(3333 4444 5555 6666 7777 8888 9999)
ITERS=(4 5 6 7 8 9 10)
FRAMES=10

echo "=============================================="
echo " Iterative Verification: iter ${ITERS[0]}-${ITERS[-1]}"
echo " Frames per iter: $FRAMES"
echo "=============================================="

ALL_PASS=true
for i in "${!ITERS[@]}"; do
    iter=${ITERS[$i]}
    seed=${SEEDS[$i]}
    outdir="$BASE_DIR/test_iter${iter}"

    echo ""
    echo "--- Iter $iter (seed=$seed, $FRAMES frames) ---"

    # 생성
    python "$GEN_SCRIPT" \
        --num_frames "$FRAMES" \
        --output_dir "$outdir" \
        --seed "$seed" \
        --overlay

    # 검증
    echo ""
    echo "--- Verify iter $iter ---"
    if python "$VERIFY_SCRIPT" "$outdir"; then
        echo "[iter $iter] ALL PASS"
    else
        echo "[iter $iter] FAIL"
        ALL_PASS=false
    fi

    echo ""
done

echo "=============================================="
if $ALL_PASS; then
    echo "RESULT: ALL ITERATIONS PASSED"
else
    echo "RESULT: SOME ITERATIONS FAILED"
fi
echo "=============================================="
