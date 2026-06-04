@echo off
REM YOLO26n-pose v1 ft_manual — padded manual GT 219, 60 ep, lr 1e-4
REM project 는 absolute path 로 (runs/pose/ misroute 회피)

call C:\Users\minjae\anaconda3\Scripts\activate.bat pallet-pose
cd /d C:\Users\minjae\Documents\github\FoundationPose
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE

yolo pose train data=challenge/yolo_pose/data_manual_padded.yaml model=challenge/weights/yolo26n_pose_v1/weights/best.pt optimizer=SGD lr0=0.0001 epochs=60 imgsz=640 batch=16 project=C:/Users/minjae/Documents/github/FoundationPose/challenge/weights name=yolo26n_pose_v1_ft_manual device=0 workers=4 patience=30 save_period=10 exist_ok=True plots=True 1>>challenge/weights/yolo26n_pose_v1_ft_manual_detached.log 2>&1
