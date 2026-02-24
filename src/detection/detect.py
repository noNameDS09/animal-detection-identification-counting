from ultralytics import YOLO

model = YOLO(r"../../notebooks/runs/detect/train/weights/best.pt")

model.predict(
    source="../../data/raw_videos/elephant.mp4",
    show=True,
    conf=0.5
)