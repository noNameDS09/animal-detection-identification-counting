"""
Animal Counter using YOLO Object Detection + Tracking

This script processes a video file, detects and tracks animals using a YOLO model,
and counts unique instances of each detected species.

Features
-
- YOLO object detection
- ByteTrack multi-object tracking
- Unique animal counting
- Minimum frame validation to reduce false positives
- Optional region-based counting
- Real-time visualization
- Processing FPS display

"""

import time
import logging
from collections import defaultdict
from typing import Tuple, Dict, Set

import cv2
import torch
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.reporting.report_generator import ReportGenerator


# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



# AnimalCounter Class

class AnimalCounter:
    """
    Detects, tracks, and counts unique animals in a video.

    Parameters:    
        model_path : str
            Path to trained YOLO model.
        video_path : str
            Path to input video.
        conf : float
            Detection confidence threshold.
        min_frames : int
            Minimum number of frames an object must appear before being counted.
        resize_dim : Tuple[int, int], optional
            Frame resize dimensions for processing. If None, original dimensions are used (default: (1280, 720)).
        show_boxes : bool
            Whether to show bounding boxes (default: True).
        show_labels : bool
            Whether to show class labels (default: True).
        show_conf : bool
            Whether to show confidence scores (default: True).
        show_counts : bool
            Whether to show dynamic counts overlay (default: True).
    """

    def __init__(
        self,
        model_path: str,
        video_path: str,
        conf: float = 0.5,
        min_frames: int = 5,
        resize_dim: Tuple[int, int] = None,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_counts: bool = True,
    ) -> None:

        self.model_path = model_path
        self.video_path = video_path
        self.conf = conf
        self.min_frames = min_frames
        self.resize_dim = resize_dim
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.show_counts = show_counts

        # Determine compute device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        self.model = self._load_model()

        # Initialize video capture
        self.cap = self._open_video()

        # If resize_dim is None, use original video dimensions
        if self.resize_dim is None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resize_dim = (width, height)
            logger.info(f"Using original video dimensions: {self.resize_dim}")

        # Video FPS for display timing
        self.video_fps = self._get_video_fps()
        self.delay = int(1000 / self.video_fps)

        # Tracking statistics
        self.id_frame_count: Dict[int, int] = defaultdict(int)
        self.unique_ids: Dict[str, Set[int]] = defaultdict(set)
        
        # Reporting statistics
        self.appearance_times: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.frame_data = [] # List of dicts for frame-wise CSV


    def _load_model(self) -> YOLO:
        """
        Load YOLO model and move it to appropriate device.

        Returns::
            Loaded YOLO model.
        """
        logger.info("Loading YOLO model...")

        model = YOLO(self.model_path)
        model.model.float()
        model.to(self.device)

        logger.info(f"Model loaded on device: {self.device}")

        return model

    

    def _open_video(self) -> cv2.VideoCapture:
        """
        Open input video file.
        Returns::
            cv2.VideoCapture

        """

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        logger.info("Video opened successfully.")

        return cap

    

    def _get_video_fps(self) -> float:
        """
        Retrieve FPS from video file.

        Returns:
            float
                Frames per second.
        
        """

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            logger.warning("FPS detection failed. Defaulting to 60 FPS.")
            fps = 60.0

        logger.info(f"Video FPS: {fps}")

        return fps

    

    def _process_frame(self, frame):
        """
        Perform detection, tracking, and counting on a single frame.

        Parameters:
            frame : ndarray
                Video frame.

        Returns:
            ndarray
                Annotated frame.
        
        """

        frame = cv2.resize(frame, self.resize_dim)

        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            tracker="bytetrack.yaml",
            imgsz=640
        )

        if results[0].boxes.id is not None:

            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for obj_id, cls_id in zip(ids, classes):

                self.id_frame_count[obj_id] += 1

                if self.id_frame_count[obj_id] >= self.min_frames:

                    class_name = self.model.names[int(cls_id)]

                    if int(obj_id) not in self.unique_ids[class_name]:
                        self.unique_ids[class_name].add(int(obj_id))
                        # Record first appearance timestamp
                        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        timestamp_sec = current_frame / self.video_fps if self.video_fps > 0 else 0
                        self.appearance_times[class_name][int(obj_id)] = round(timestamp_sec, 2)
                        
            # Record frame-wise summary
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp_sec = current_frame / self.video_fps if self.video_fps > 0 else 0
            
            frame_summary = {
                'frame': current_frame,
                'timestamp_sec': round(timestamp_sec, 2)
            }
            # Fill species count for this frame
            for cls_id in classes:
                class_name = self.model.names[int(cls_id)]
                if class_name not in frame_summary:
                    frame_summary[class_name] = 0
                frame_summary[class_name] += 1
            
            self.frame_data.append(frame_summary)

        annotated_frame = results[0].plot(
            boxes=self.show_boxes,
            labels=self.show_labels,
            conf=self.show_conf
        )

        return annotated_frame

    

    def _draw_overlay(self, frame, processing_fps: float):
        """
        Draw counting statistics on frame.

        Parameters:
            frame : ndarray
            processing_fps : float

        Returns:
            ndarray
        
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
                0.5,
                (0, 255, 0),
                1
            )

            y_offset += 30

        # Total count
        cv2.putText(
            frame,
            f"Total Unique Animals: {total_count}",
            (20, y_offset + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            1
        )

        # Processing FPS
        cv2.putText(
            frame,
            f"Processing FPS: {processing_fps:.2f}",
            (20, y_offset + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            1
        )

        return frame

    

    def run(self) -> None:
        """
        Main processing loop.
        """

        logger.info("Starting video processing...")

        cv2.namedWindow("Animal Counter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Animal Counter", *self.resize_dim)

        # Add Trackbar
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar("Progress", "Animal Counter", 0, total_frames, lambda x: None)

        is_paused = False
        last_frame_rendered = None

        while True:
            # Handle manual seek via trackbar
            trackbar_pos = cv2.getTrackbarPos("Progress", "Animal Counter")
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            if abs(trackbar_pos - current_frame) > 5:
                # User scrubbed the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_pos)
                logger.warning("Video scrubbed: Tracker may lose previous objects and recounts might occur.")
                
                # Render a single frame to update window while paused
                ret, frame = self.cap.read()
                if ret:
                    start_time = time.time()
                    annotated_frame = self._process_frame(frame)
                    processing_fps = 1 / (time.time() - start_time)
                    if self.show_counts:
                        annotated_frame = self._draw_overlay(annotated_frame, processing_fps)
                    last_frame_rendered = annotated_frame
                    cv2.setTrackbarPos("Progress", "Animal Counter", int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

            if not is_paused:
                ret, frame = self.cap.read()

                if not ret:
                    logger.info("End of video reached.")
                    break

                start_time = time.time()
                annotated_frame = self._process_frame(frame)

                processing_fps = 1 / (time.time() - start_time)

                if self.show_counts:
                    annotated_frame = self._draw_overlay(annotated_frame, processing_fps)
                
                last_frame_rendered = annotated_frame
                cv2.setTrackbarPos("Progress", "Animal Counter", int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

            # Update Window
            if last_frame_rendered is not None:
                display_frame = last_frame_rendered.copy()
                if is_paused:
                    cv2.putText(
                        display_frame,
                        "PAUSED (Space = Play/Pause | Q = Quit)",
                        (20, self.resize_dim[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                cv2.imshow("Animal Counter", display_frame)

            wait_time = max(1, self.delay) if not is_paused else 50
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord("q"):
                logger.info("Processing interrupted by user.")
                break
            elif key == ord(" "):
                is_paused = not is_paused
                logger.info(f"Video {'paused' if is_paused else 'resumed'}.")

        self._cleanup()

    

    def _cleanup(self) -> None:
        """
        Release resources, print final statistics, and generate reports.

        """

        self.cap.release()
        cv2.destroyAllWindows()

        logger.info("\nFinal Unique Animal Counts:")

        for species, ids in self.unique_ids.items():
            logger.info(f"{species}: {len(ids)}")
            
        # Generate reports
        try:
            reporter = ReportGenerator(output_dir="outputs")
            import os
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            
            reporter.generate_framewise_csv(video_name, self.frame_data)
            reporter.generate_video_summary(video_name, self.unique_ids, self.appearance_times)
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")



# Entry Point

if __name__ == "__main__":

    counter = AnimalCounter(
        model_path=r"../../models/trained/best_10000_images.pt",
        video_path=r"../../data/raw_videos/elephanttigercheetah.mp4",
        conf=0.5,
        min_frames=5,
        resize_dim=None,
        show_boxes=True,
        show_labels=True,
        show_conf=True,
        show_counts=True,
    )

    counter.run()