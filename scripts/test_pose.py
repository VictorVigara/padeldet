from ultralytics import YOLO

model = YOLO("yolo26m-pose.pt")
# model = YOLO("yolov8n.pt")
results = model.predict("/home/vvipu/padeldet/data/yt/2/2.mp4", show=True, conf = 0.0001, imgsz=960)