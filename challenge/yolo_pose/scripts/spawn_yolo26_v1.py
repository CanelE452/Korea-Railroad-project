"""YOLO26 v1 wrapper 를 Windows DETACHED_PROCESS 로 spawn."""
import subprocess
import os

WRAPPER = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\yolo_pose\scripts\auto_restart_yolo26_v1.py"
PYTHON  = r"C:\Users\minjae\anaconda3\python.exe"

DETACHED_PROCESS          = 0x00000008
CREATE_NEW_PROCESS_GROUP  = 0x00000200
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
print(f"YOLO26_V1_WRAPPER_PID={proc.pid}")
