import time
from collections import defaultdict
from typing import Tuple

import cv2
import torch
from ultralytics import YOLO


class AnimalCounter:
    """
    AnimalCounter performs object tracking on a video using a YOLO model
    and counts unique tracked instances per class.

    Attributes:
        model_path (str): Path to trained YOLO model.
        video_path (str): Path to input video.
        conf (float): Detection confidence threshold.
        min_frames (int): Minimum frames an ID must appear to be counted.
        resize_dim (Tuple[int, int]): Frame resize dimensions.
    """

    def __init__(
        self,
        model_path: str,
        video_path: str,
        conf: float = 0.5,
        min_frames: int = 5,
        resize_dim: Tuple[int, int] = (1280, 720),
    ) -> None:

        self.model_path = model_path
        self.video_path = video_path
        self.conf = conf
        self.min_frames = min_frames
        self.resize_dim = resize_dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._initialize_model()

        self.cap = cv2.VideoCapture(self.video_path)
        self.video_fps = self._get_video_fps()
        self.delay = int(1000 / self.video_fps)

        self.id_frame_count = defaultdict(int)
        self.unique_ids = defaultdict(set)

    def _initialize_model(self) -> YOLO:
        """
        Loads the YOLO model and prepares it for inference.
        """
        model = YOLO(self.model_path)
        model.model.float()
        model.to(self.device)

        print(f"Running on device: {self.device}")
        return model

    def _get_video_fps(self) -> float:
        """
        Retrieves FPS from the video file.
        Falls back to 60 FPS if detection fails.
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 60.0

        print(f"Video FPS: {fps}")
        return fps

    def _track_and_count(self, frame):
        """
        Performs tracking on a frame and updates unique object counts.
        """
        frame = cv2.resize(frame, self.resize_dim)

        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            imgsz=640,
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for obj_id, cls_id in zip(ids, classes):
                self.id_frame_count[obj_id] += 1

                if self.id_frame_count[obj_id] >= self.min_frames:
                    class_name = self.model.names[int(cls_id)]
                    self.unique_ids[class_name].add(int(obj_id))

        return results[0].plot()

    def _draw_overlay(self, frame, processing_fps: float):
        """
        Draws counting statistics and FPS information on the frame.
        """
        y_offset = 30
        total_count = 0

        for species in self.model.names.values():
            count = len(self.unique_ids[species])
            total_count += count

            cv2.putText(
                frame,
                f"{species}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += 30

        cv2.putText(
            frame,
            f"Total Unique Animals: {total_count}",
            (20, y_offset + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Processing FPS: {processing_fps:.2f}",
            (20, y_offset + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        return frame

    def run(self) -> None:
        """
        Executes the video processing loop.
        """
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Output", *self.resize_dim)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            start_time = time.time()

            annotated_frame = self._track_and_count(frame)
            processing_fps = 1 / (time.time() - start_time)

            annotated_frame = self._draw_overlay(
                annotated_frame,
                processing_fps,
            )

            cv2.imshow("Output", annotated_frame)

            if cv2.waitKey(self.delay) & 0xFF == ord("q"):
                break

        self._cleanup()

    def _cleanup(self) -> None:
        """
        Releases resources and prints final counts.
        """
        self.cap.release()
        cv2.destroyAllWindows()

        print("\nFinal Unique Animal Count:")
        for species, ids in self.unique_ids.items():
            print(f"{species}: {len(ids)}")


if __name__ == "__main__":
    counter = AnimalCounter(
        model_path="D:/TE/Internship/code/models/trained/best_10000_images.pt",
        video_path="D:/TE/Internship/code/data/raw_videos/elephanthd.mp4",
        conf=0.5,
        min_frames=5,
        resize_dim=(1280, 720),
    )

    counter.run()