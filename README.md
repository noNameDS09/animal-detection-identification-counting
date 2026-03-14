# Animal Detection, Identification and Counting

This repository contains the codebase for the **Animal Detection, Identification, and Counting** internship assignment.

The system relies on an Ultralytics YOLOv26 object detection model fine-tuned on a custom dataset, integrated with ByteTrack to assign unique identifiers and count species appearing in video footage without duplicate counts.

Following are the animals detected by the system:

- Elephant
- Tiger
- Deer
- Monkey
- Bear
- Lion
- Leopard
- Cheetah
- Jaguar
- Zebra
- Giraffe
- Hippopotamus
- Rhinoceros
- Wolf
- Fox
- Hyena
- Gorilla
- Chimpanzee
- Baboon
- Crocodile
- Alligator
- Snake
- Lizard
- Kangaroo
- Panda
- Otter
- Squirrel

## Features

- **Object Detection:** Detects animals in video streams using YOLO.
- **Multi-Object Tracking:** Assigns unique IDs to animals using ByteTrack to prevent duplicate counting during occlusion or movement.
- **Automated Reporting:** Outputs a frame-wise presence CSV and a final summary report (JSON and CSV) recording total unique animals and their first appearance timestamp.


## Installation

1. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the main Animal Counter on a video file:

```bash
python src/counting/counter.py
```

_Note: You may need to edit `counter.py`'s `__init__` block at the bottom of the file to change the `video_path` or `model_path` pointers depending on your required inputs._

Once the video finishes (or is interrupted by pressing `q`), the generation scripts will automatically output the CSV and JSON reports into the `outputs/` folder.



## Project Structure

- `data/` - Raw videos, extracted frames, and annotation datasets.
- `docs/` - Requirements spec, test matrices, and improvement roadmaps.
- `models/` - Saved `.pt` weights from custom YOLO training.
- `notebooks/` - Jupyter notebooks for baseline evaluation and dataset structuring.
- `outputs/` - Generated JSON and CSV reports.
- `src/` - Source code for detection, tracing, counting, and reporting.
- `tests/` - Pytest unit and integration scripts.
