@echo off
REM YOLOv8n-pose v2 resume from last.pt to epoch 100
REM 한 줄로 작성 (line continuation ^ 가 line-ending 차이로 escape 되는 버그 회피)

call C:\Users\minjae\anaconda3\Scripts\activate.bat pallet-pose
cd /d C:\Users\minjae\Documents\github\FoundationPose
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE

yolo pose train resume=True model=challenge/weights/yolov8n_pose_v2/weights/last.pt project=challenge/weights name=yolov8n_pose_v2 exist_ok=True 1>>challenge/weights/yolov8n_pose_v2_train_detached.log 2>&1
