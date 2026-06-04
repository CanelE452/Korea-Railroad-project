"""Manual GT fine-tune auto-restart wrapper.

ft_manual.bat 을 실행하고, 죽으면 30초 대기 후 자동 재시작.
종료 조건: results.csv 의 last epoch >= 60 또는 max restarts 도달.
"""
import subprocess
import time
import os
import csv

RESULTS_CSV = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\weights\yolov8n_pose_v2_ft_manual\results.csv"
TRAIN_BAT   = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\yolo_pose\scripts\ft_manual.bat"
WRAPPER_LOG = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\weights\yolov8n_pose_v2_ft_manual_auto_restart.log"

MAX_EPOCH = 60
MAX_RESTARTS = 10
RESTART_DELAY = 30


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    print(line, end="", flush=True)
    with open(WRAPPER_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def last_epoch():
    if not os.path.exists(RESULTS_CSV):
        return 0
    try:
        with open(RESULTS_CSV) as f:
            rows = list(csv.DictReader(f))
        return int(rows[-1]["epoch"]) if rows else 0
    except Exception as e:
        log(f"failed to read csv: {e}")
        return 0


def main():
    log("=" * 60)
    log(f"FT auto-restart wrapper started (PID {os.getpid()})")
    log(f"Target: epoch {MAX_EPOCH}, max restarts {MAX_RESTARTS}")

    restart = 0
    while True:
        ep = last_epoch()
        log(f"Current epoch: {ep}/{MAX_EPOCH}")
        if ep >= MAX_EPOCH:
            log(f"TARGET REACHED (epoch {ep} >= {MAX_EPOCH}). Exiting.")
            break
        if restart >= MAX_RESTARTS:
            log(f"MAX RESTARTS reached ({MAX_RESTARTS}). Exiting.")
            break

        restart += 1
        log(f"Starting ft attempt {restart}...")
        t0 = time.time()
        try:
            rc = subprocess.run(["cmd.exe", "/c", TRAIN_BAT]).returncode
        except Exception as e:
            log(f"subprocess failed: {e}")
            rc = -1
        dt = time.time() - t0
        new_ep = last_epoch()
        log(f"FT exited rc={rc} after {dt/60:.1f} min, epoch {ep} -> {new_ep}")

        if new_ep >= MAX_EPOCH:
            log(f"TARGET REACHED after restart. Exiting.")
            break

        if new_ep == ep and dt < 60:
            log(f"WARNING: no progress + crashed in {dt:.0f}s. Sleeping longer (5min)...")
            time.sleep(300)
        else:
            log(f"Sleeping {RESTART_DELAY}s before restart...")
            time.sleep(RESTART_DELAY)

    log("Wrapper finished.")


if __name__ == "__main__":
    main()
