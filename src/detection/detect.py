from ultralytics import YOLO

def main():
    model = YOLO(r"D:\TE\Internship\code\models\trained\best_10000_images.pt")

    metrics = model.val(
        data=r"D:\TE\Internship\code\data\dataset\v2_10000_images\dataset.yaml",
        imgsz=640,
        batch=16,
        device=0
    )

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("F1:", metrics.box.f1)

if __name__ == "__main__":
    main()