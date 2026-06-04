@echo off
REM YOLOv8n-pose v2_ft_manual fine-tune
REM v2 best.pt → padded manual 219 frame, 60 ep, lr 1e-4

call C:\Users\minjae\anaconda3\Scripts\activate.bat pallet-pose
cd /d C:\Users\minjae\Documents\github\FoundationPose
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE

yolo pose train data=challenge/yolo_pose/data_manual_padded.yaml model=challenge/weights/yolov8n_pose_v2/weights/best.pt optimizer=SGD lr0=0.0001 epochs=60 imgsz=640 batch=16 project=challenge/weights name=yolov8n_pose_v2_ft_manual patience=30 save_period=10 exist_ok=True plots=True 1>>challenge/weights/yolov8n_pose_v2_ft_manual_detached.log 2>&1
