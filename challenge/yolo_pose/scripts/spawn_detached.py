"""학습 wrapper 를 Windows DETACHED_PROCESS 로 spawn (부모와 완전 분리).

auto_restart_train.py 를 detached 로 실행 →
   학습 죽어도 wrapper 가 자동 재시작
   wrapper 자체도 부모 (Bash tool) 와 완전 독립

usage: python spawn_detached.py
"""
import subprocess
import os

# auto_restart_train.py 의 경로
WRAPPER = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\yolo_pose\scripts\auto_restart_train.py"
PYTHON  = r"C:\Users\minjae\anaconda3\python.exe"

# Windows process creation flags
DETACHED_PROCESS         = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB

proc = subprocess.Popen(
    [PYTHON, WRAPPER],
    creationflags=flags,
    close_fds=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"WRAPPER_PID={proc.pid}")
