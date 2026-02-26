from ultralytics import YOLO

model = YOLO(r"D:\TE\Internship\code\models\trained\best_10000_images.pt")

model.predict(
    source="../../data/raw_videos/elephant.mp4",
    show=True,
    conf=0.5
)