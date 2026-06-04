@echo off
title Annotate forklift_20260528
call C:\Users\minjae\anaconda3\Scripts\activate.bat pallet-pose
cd /d C:\Users\minjae\Documents\github\FoundationPose
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
python challenge\scripts\annotate.py --seq data/outside/forklift_raw_20260528_163408 --out_dir challenge/data/forklift_20260528_manual_gt --stride 30
echo.
echo === annotate.py exited ===
pause
