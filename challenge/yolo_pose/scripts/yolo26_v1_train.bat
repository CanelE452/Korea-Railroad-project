@echo off
REM YOLO26n-pose v1 — padded synthetic scratch, 100 ep

call C:\Users\minjae\anaconda3\Scripts\activate.bat pallet-pose
cd /d C:\Users\minjae\Documents\github\FoundationPose
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE

yolo pose train data=challenge/yolo_pose/data_padded.yaml model=yolo26n-pose.pt epochs=100 imgsz=640 batch=32 project=C:/Users/minjae/Documents/github/FoundationPose/challenge/weights name=yolo26n_pose_v1 device=0 workers=4 patience=30 save_period=10 exist_ok=True plots=True 1>>challenge/weights/yolo26n_pose_v1_train_detached.log 2>&1
