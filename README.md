# Emotional Detection (Face Landmarker)

Small demo that uses MediaPipe Face Landmarker to infer a simple emotion label (happy, surprised, sad, angry, neutral) from face blendshape outputs and shows a matching example image.

## What this project does

- Captures live webcam frames using OpenCV.
- Runs MediaPipe's Face Landmarker (live-stream mode) to compute facial landmarks and face blendshapes.
- Heuristically maps blendshape scores to one of five emotion categories using the function `infer_emotion_from_blendshapes` in `main.py`.
- Displays the detected emotion and a representative image from the `face/` folder.

## Files of interest

- `main.py` — main application and inference logic.
- `face_landmarker.task` — MediaPipe model asset required by the Face Landmarker. Keep this file in the project root (or update `model_path` in `main.py`).
- `face/` — contains sample images used to show the inferred emotion (e.g. `happy.jpg`, `sad.jpg`, etc.).

## Requirements

- macOS / Linux / Windows with a webcam
- Python 3.8+
- The Python packages listed in `requirements.txt` (install with pip)

Recommended packages (also included in `requirements.txt`):

- mediapipe
- opencv-python

## Install

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you run into problems installing `mediapipe` on macOS, check the official MediaPipe installation notes for your platform or use a pre-built wheel when appropriate.

## Run

Make sure your webcam is connected and accessible. Then run:

```bash
python main.py
```

Controls:

- The application runs until you press `Esc` (escape key).
- Two windows will open: one showing the live camera with the landmarker overlay, and one showing the representative emotion image.

## Algorithm / How emotion is inferred

The project does not use a trained classifier for emotion. Instead it maps MediaPipe face blendshape values (per-frame float scores in the range [0..1]) to a simple heuristic scoring function implemented in `infer_emotion_from_blendshapes` in `main.py`.

Key points about the heuristic:

- It aggregates left/right blendshape scores (e.g. `mouthSmileLeft`, `mouthSmileRight`) into composite signals like `smile`, `frown`, `eye_wide`, `jaw_open`, `brow_down`, etc.
- Each emotion (happy, surprised, angry, sad, neutral) gets a weighted sum formed from these signals.
- A confidence gating step compares the top two scores and if the margin or absolute score is weak, the result falls back to `neutral`.

This makes the demo readable and easy to tweak—weights are in `main.py` and can be tuned for your camera, lighting, or dataset.

## Configuration

- `model_path` in `main.py` points at `face_landmarker.task`. If you keep the model at a different path, update that variable.
- `face_images` in `main.py` maps emotion keys to image files. Replace images with your own assets if desired.

## Troubleshooting

- No camera detected / OpenCV can't open camera: ensure no other application is using the webcam and that you granted camera permission to the terminal / Python process.
- `mediapipe` install fails: check Python version and platform; on some macOS setups you may need to install additional build tools or use a wheel matching your Python.
- Low or incorrect detection: lighting and camera angle can greatly influence blendshape scores. Try even lighting, facing the camera, and moderate distance.

## Extending the project

- Replace the heuristic with a small classifier trained on labeled blendshapes if you need higher accuracy.
- Log blendshape outputs and build a dataset to train a supervised model.
- Support multiple faces (`num_faces` option in `main.py`) and visualize scores on-screen.
