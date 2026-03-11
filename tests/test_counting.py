import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.counting.counter import AnimalCounter

@pytest.fixture
def sample_video_path():
    """Returns the path to a real sample video from the data directory."""
    path = os.path.join("data", "raw_videos", "elephanttigercheetah.mov")
    if not os.path.exists(path):
        pytest.skip(f"Test video not found at {path}. Please ensure it exists before running integration tests.")
    return path
    
@pytest.fixture
def model_path():
    """Returns the path to the trained YOLO model."""
    path = os.path.join("models", "trained", "best_10000_images.pt")
    if not os.path.exists(path):
        pytest.skip(f"Trained model not found at {path}. Please ensure it exists before running integration tests.")
    return path

def test_actual_counting_integration_loop(sample_video_path, model_path):
    """
    Test running the AnimalCounter on a real video for a limited number of frames.
    This ensures OpenCV opens the video, YOLO performs detection, and the ByteTracker
    assigns IDs and increments frame counts correctly.
    """
    counter = AnimalCounter(
        model_path=model_path,
        video_path=sample_video_path,
        conf=0.25, # Lower confidence for testing to ensure we capture something
        min_frames=2, # Require only 2 frames so we can test quickly
        show_boxes=False, # Disable UI for automated test
        show_labels=False,
        show_conf=False,
        show_counts=False
    )
    
    # Manually run the processing loop for a set number of test frames
    test_frames = 15
    frames_processed = 0
    
    while frames_processed < test_frames:
        ret, frame = counter.cap.read()
        if not ret:
            break
            
        counter._process_frame(frame)
        frames_processed += 1
        
    # We expect that the counter successfully read the frames and tracked some objects
    assert frames_processed > 0, "No frames were processed from the video"
    
    # We expect some tracking data was recorded (this depends on the video, 
    # but the elephanttigercheetah video should have animals in the first 15 frames)
    # If this fails, the test suite might need a video with objects closer to the start,
    # or the test_frames should be increased.
    has_tracks = len(counter.id_frame_count) > 0
    assert has_tracks, "No objects were tracked in the first 15 frames"
    
    # Clean up the windows and release video capture implicitly (without triggering the final report)
    counter.cap.release()

