# Phase 7: Test Report & Validation Matrix

## 1. Test Matrix

The following table documents the model's performance under various real-world conditions based on our testing dataset.

| Scene Type               | Conditions                               | Model Performance & Observations                                                                                                                                                                     |
| :----------------------- | :--------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Daytime Footage**      | Clear lighting, unobstructed views       | **Excellent.** High confidence bounding boxes (>0.75). Tracking IDs persist well across the camera's frame.                                                                                          |
| **Low-light Footage**    | Dusk/Dawn, shadows, limited visibility   | **Fair to Good.** Detection confidence drops. Some dark-colored species (e.g., elephants) may merge with shadows resulting in missed detections or split tracking IDs if they momentarily disappear. |
| **Dense Animal Groups**  | Herds, overlapping bodies (e.g., Zebras) | **Moderate.** YOLO detects the herd, but ByteTrack occasionally struggles to maintain unique IDs when animals cross in front of one another (occlusion).                                             |
| **Moving Camera scenes** | Handheld panning, vehicle-mounted        | **Good.** ByteTrack handles linear camera motion well, but abrupt jitters or fast panning can cause ID switching, resulting in slightly inflated unique animal counts.                               |

---

## 2. Documented Limitations

Throughout the development and baseline testing phases, the following limitations were observed:

1. **Occlusion Handling:** When a large animal completely obscures a smaller animal, the tracker drops the hidden animal's ID. When the smaller animal re-emerges, it is occasionally assigned a _new_ unique ID, which leads to double counting.
2. **Camera Jitter:** Fast erratic movements of the camera (often seen in documentary or amateur wildlife footage) cause motion blur. This briefly drops bounding boxes, breaking the tracking continuity.


---

## 3. Improvement Roadmap

To enhance the system for future iterations or production deployment, the following improvements are planned:

- **Short-term Improvements:**
  - **Increase `min_frames` dynamically:** Adjust the minimum frame appearance threshold based on the video's FPS to filter out ghost detections (false positives that appear for a fraction of a second).
  - **Implement Re-ID (Appearance Tracking):** Upgrade from standard ByteTrack (which relies primarily on bounding box physics/IoU) to DeepSORT or an advanced Re-ID model (e.g., BoT-SORT) that uses visual features to reconnect animals that were temporarily occluded.
- **Long-term Improvements:**
  - **Model Quantization:** Convert the `.pt` weights to `.engine` (TensorRT) or `.onnx` for faster inference on edge devices (like field cameras).
  - **Region of Interest (ROI) Counting Line:** Instead of counting every unique ID that appears anywhere on the screen, implement a virtual "tripwire" line. Animals are only counted once they cross this line, which drastically improves accuracy for directional migration videos.
