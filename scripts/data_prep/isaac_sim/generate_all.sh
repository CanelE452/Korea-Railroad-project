#!/bin/bash
# Pallet Synthetic Data Generation - Batch Script
#
# 64프레임마다 Isaac Sim을 재시작하여 메모리 누수 방지.
# v11: 2,000 train + 500 val (리프터 마운트 60%, 야외 60%, 적재물 80%)
# 기존 데이터(train_batch_*)는 건드리지 않음.
# 완료된 배치는 자동 건너뜀 (resume 지원).

# set -e 제거: Isaac Sim 종료 시 segfault(139)는 비치명적이므로 무시

# 잔여 Isaac Sim (gen_replicator_data.py) 프로세스 정리
STALE_PIDS=$(wmic process where "name='python.exe' and CommandLine like '%gen_replicator_data%'" get ProcessId 2>/dev/null | grep -oE '[0-9]+')
if [ -n "$STALE_PIDS" ]; then
    echo "[WARN] 잔여 Isaac Sim 프로세스를 정리합니다..."
    for pid in $STALE_PIDS; do
        taskkill //F //PID "$pid" 2>/dev/null
    done
    sleep 5
fi

# Lockfile 기반 중복 실행 방지
LOCKFILE="/tmp/generate_all.lock"
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] generate_all.sh가 이미 실행 중입니다 (PID: $OLD_PID). 종료합니다."
        exit 1
    else
        echo "[WARN] stale lockfile 제거 (PID $OLD_PID 이미 종료됨)"
        rm -f "$LOCKFILE"
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="C:/Users/minjae/anaconda3/envs/pallet-pose/python.exe"
SCRIPT="scripts/data_prep/isaac_sim/gen_replicator_data.py"
TRAIN_BASE="data/pallet/training_data"
BATCH_SIZE=64
RENDERER="PathTracing"
HDRI_DIR="data/pallet/hdri"

# v11: 출력 디렉토리 접두사 (기존 train_batch_* 보존)
BATCH_PREFIX="${BATCH_PREFIX:-train_v11_batch}"
VAL_PREFIX="${VAL_PREFIX:-val_v11_batch}"
# seed offset (기존 배치와 겹치지 않도록)
TRAIN_SEED_BASE="${TRAIN_SEED_BASE:-5000}"
VAL_SEED_BASE="${VAL_SEED_BASE:-15000}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export CUDA_MODULE_LOADING=LAZY

TRAIN_TOTAL=2000
VAL_TOTAL=500
TRAIN_BATCHES=200   # 충분히 크게 (실제 프레임 수로 제어)
VAL_BATCHES=50      # 충분히 크게 (실제 프레임 수로 제어)

# 배치 완료 확인 함수: PNG/JSON이 최소 10개 이상이면 완료로 간주
# (이전 10-frame 배치 + 현재 20-frame 배치 모두 호환)
is_batch_done() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    local png_count=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    local json_count=$(ls "$dir"/*.json 2>/dev/null | wc -l)
    [ "$png_count" -ge 10 ] && [ "$json_count" -ge 10 ]
}

echo "============================================"
echo " Pallet Synthetic Data Generation"
echo "============================================"
echo " Renderer:    $RENDERER"
echo " Batch size:  $BATCH_SIZE frames per restart"
echo " Train:       $TRAIN_TOTAL frames ($TRAIN_BATCHES batches)"
echo " Val:         $VAL_TOTAL frames ($VAL_BATCHES batches)"
echo " Total:       $((TRAIN_TOTAL + VAL_TOTAL)) frames"
echo "============================================"

# 이미 완료된 프레임 수 카운트 (실제 PNG 기준)
SKIPPED_TRAIN=0
SKIPPED_TRAIN_FRAMES=0
SKIPPED_VAL=0
SKIPPED_VAL_FRAMES=0
for i in $(seq 0 $((TRAIN_BATCHES - 1))); do
    dir="$TRAIN_BASE/${BATCH_PREFIX}_${i}"
    if is_batch_done "$dir"; then
        SKIPPED_TRAIN=$((SKIPPED_TRAIN + 1))
        SKIPPED_TRAIN_FRAMES=$((SKIPPED_TRAIN_FRAMES + $(ls "$dir"/*.png 2>/dev/null | wc -l)))
    fi
done
for i in $(seq 0 $((VAL_BATCHES - 1))); do
    dir="$TRAIN_BASE/${VAL_PREFIX}_${i}"
    if is_batch_done "$dir"; then
        SKIPPED_VAL=$((SKIPPED_VAL + 1))
        SKIPPED_VAL_FRAMES=$((SKIPPED_VAL_FRAMES + $(ls "$dir"/*.png 2>/dev/null | wc -l)))
    fi
done
echo " Already done: train=$SKIPPED_TRAIN batches ($SKIPPED_TRAIN_FRAMES frames), val=$SKIPPED_VAL batches ($SKIPPED_VAL_FRAMES frames)"
echo "============================================"

TOTAL_TRAIN=$SKIPPED_TRAIN_FRAMES

for i in $(seq 0 $((TRAIN_BATCHES - 1))); do
    SEED=$((TRAIN_SEED_BASE + i * 50))
    BATCH_DIR="$TRAIN_BASE/${BATCH_PREFIX}_${i}"

    # 이미 완료된 배치 건너뛰기
    if is_batch_done "$BATCH_DIR"; then
        echo "[SKIP] ${BATCH_PREFIX}_$i already done ($(ls "$BATCH_DIR"/*.png 2>/dev/null | wc -l) PNG)"
        continue
    fi

    REMAINING=$((TRAIN_TOTAL - TOTAL_TRAIN))
    if [ "$REMAINING" -le 0 ]; then break; fi
    N_FRAMES=$BATCH_SIZE
    if [ "$REMAINING" -lt "$N_FRAMES" ]; then N_FRAMES=$REMAINING; fi

    echo ""
    echo "[TRAIN] ${BATCH_PREFIX}_$i (seed=$SEED, $N_FRAMES frames) -> $BATCH_DIR"

    # 최대 3회 시도 (크래시 시 재시도)
    for attempt in 1 2 3; do
        BATCH_LOG="$BATCH_DIR.log"
        "$PYTHON_EXE" "$SCRIPT" \
            --renderer "$RENDERER" \
            --num_frames "$N_FRAMES" \
            --output_dir "$BATCH_DIR" \
            --seed "$SEED" \
            --hdri_dir "$HDRI_DIR" \
            > "$BATCH_LOG" 2>&1

        N_DONE=$(ls "$BATCH_DIR"/*.png 2>/dev/null | wc -l)
        if [ "$N_DONE" -ge "$((N_FRAMES / 2))" ]; then
            break
        fi
        echo "[RETRY] Batch $i attempt $attempt failed ($N_DONE frames). Retrying after cooldown..."
        sleep 10
    done

    TOTAL_TRAIN=$((TOTAL_TRAIN + N_DONE))
    echo "[TRAIN] ${BATCH_PREFIX}_$i done. $N_DONE/$N_FRAMES frames. Total train: $TOTAL_TRAIN/$TRAIN_TOTAL"
    sleep 5
done

TOTAL_VAL=$SKIPPED_VAL_FRAMES

for i in $(seq 0 $((VAL_BATCHES - 1))); do
    SEED=$((VAL_SEED_BASE + i * 50))
    BATCH_DIR="$TRAIN_BASE/${VAL_PREFIX}_${i}"

    if is_batch_done "$BATCH_DIR"; then
        echo "[SKIP] ${VAL_PREFIX}_$i already done ($(ls "$BATCH_DIR"/*.png 2>/dev/null | wc -l) PNG)"
        continue
    fi

    REMAINING=$((VAL_TOTAL - TOTAL_VAL))
    if [ "$REMAINING" -le 0 ]; then break; fi
    N_FRAMES=$BATCH_SIZE
    if [ "$REMAINING" -lt "$N_FRAMES" ]; then N_FRAMES=$REMAINING; fi

    echo ""
    echo "[VAL] ${VAL_PREFIX}_$i (seed=$SEED, $N_FRAMES frames) -> $BATCH_DIR"

    for attempt in 1 2 3; do
        BATCH_LOG="$BATCH_DIR.log"
        "$PYTHON_EXE" "$SCRIPT" \
            --renderer "$RENDERER" \
            --num_frames "$N_FRAMES" \
            --output_dir "$BATCH_DIR" \
            --seed "$SEED" \
            --hdri_dir "$HDRI_DIR" \
            > "$BATCH_LOG" 2>&1

        N_DONE=$(ls "$BATCH_DIR"/*.png 2>/dev/null | wc -l)
        if [ "$N_DONE" -ge "$((N_FRAMES / 2))" ]; then
            break
        fi
        echo "[RETRY] Val batch $i attempt $attempt failed ($N_DONE frames). Retrying..."
        sleep 10
    done

    TOTAL_VAL=$((TOTAL_VAL + N_DONE))
    echo "[VAL] ${VAL_PREFIX}_$i done. $N_DONE/$N_FRAMES frames. Total val: $TOTAL_VAL/$VAL_TOTAL"
    sleep 5
done

echo ""
echo "============================================"
echo " All batches complete!"
echo " Train: $TOTAL_TRAIN frames"
echo " Val:   $TOTAL_VAL frames"
echo " Total: $((TOTAL_TRAIN + TOTAL_VAL)) frames"
echo "============================================"

# 완료 알림
NTFY_TOPIC="${NTFY_TOPIC:-minjae-pallet-pose}"
curl -s -H "Content-Type: text/plain; charset=utf-8" \
    -d "Synthetic data generation done! ${TOTAL_TRAIN} train + ${TOTAL_VAL} val frames (${BATCH_PREFIX})" \
    "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
