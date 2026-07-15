# Hand Gesture Recognition App

A web app that recognizes hand gestures from a photo — either taken live via
your browser camera or uploaded as an image. Built with **MediaPipe Hands**
for landmark detection and a **scikit-learn** classifier for gesture
classification, deployed on **Streamlit Community Cloud**.

🔗 **Live app:** https://hand-gesture-recognition-model-coqwcf8ybn9ap4hhpmzfku.streamlit.app/

## Overview

Instead of classifying raw pixels, this app extracts **21 3D hand landmarks**
per image using MediaPipe, normalizes them (translation and scale invariant),
and feeds the resulting 63-value feature vector into a trained classifier.
This approach is fast, works well on CPU, and is robust to different
backgrounds and lighting compared to a raw-pixel CNN.

## Recognized Gestures

- Palm (open hand)
- Fist
- Peace sign
- Thumbs up
- Point

Custom gestures can be added by collecting more labeled samples and retraining.

## How It Works

1. **Input** — the user takes a photo via `st.camera_input()` (browser camera)
   or uploads an image file
2. **Landmark extraction** — MediaPipe Hands detects the hand and extracts 21
   (x, y, z) landmark points
3. **Normalization** — landmarks are translated relative to the wrist and
   scaled by hand size, making predictions robust to hand position and
   distance from the camera
4. **Prediction** — the normalized landmark vector is passed to a trained
   `MLPClassifier` (scikit-learn), which outputs the predicted gesture label
5. **Result** — the predicted gesture is displayed directly in the browser

## Project Structure

```
hand-gesture-recognition-model/
├── app.py                      # Streamlit web app
├── gesture_classifier (2).pkl  # Trained scikit-learn model
├── requirements.txt            # Python dependencies
├── packages.txt                # System-level dependencies (apt)
├── runtime.txt                 # Pinned Python version
└── README.md
```

## Requirements

`requirements.txt`:

```
opencv-python-headless
mediapipe>=0.10.9,<0.10.21
streamlit
numpy
scikit-learn
joblib
```

`packages.txt` (system libraries needed by OpenCV on a headless server):

```
libgl1
```

`runtime.txt` (pins Python to a version with stable MediaPipe wheel support):

```
3.10
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`, where you can
use your local webcam directly through the browser camera prompt.

## Deploying on Streamlit Community Cloud

1. Push `app.py`, the trained model file, `requirements.txt`, `packages.txt`,
   and `runtime.txt` to a GitHub repo
2. Connect the repo at share.streamlit.io
3. Streamlit Cloud installs `packages.txt` (apt) then `requirements.txt` (pip)
   automatically on each deploy
4. Visit the generated app URL

### Deployment issues we hit and fixed along the way

- **`mediapipe` `AttributeError: module has no attribute 'solutions'`** —
  caused by MediaPipe versions ≥0.10.21 removing the legacy `solutions` API.
  Fixed by pinning `mediapipe<0.10.21`.
- **`cv2` failing to import (`libGL.so.1` missing)** — OpenCV's full package
  needs system graphics libraries that a headless cloud server doesn't have.
  Fixed by switching to `opencv-python-headless` and adding `libgl1` via
  `packages.txt`.
- **`libglib2.0-0` install failure (`libffi7` unmet dependency)** — an outdated
  package name no longer resolvable on the newer Debian base image. Fixed by
  removing it from `packages.txt`, keeping only `libgl1`.
- **Blank/buffering page after a successful deploy** — the original `app.py`
  was actually a command-line script (`argparse`-based, expecting an
  `--image_path` argument), with no Streamlit UI code at all. It crashed
  immediately with no arguments provided. Fixed by rewriting it as a proper
  Streamlit app using `st.camera_input()` for browser-based camera access
  instead of a local file path.

## Known Limitations

- `st.camera_input()` captures **one still photo at a time**, not continuous
  live video — for true real-time, frame-by-frame gesture control, the
  pipeline would need to run locally with `cv2.VideoCapture(0)` instead.
- Accuracy depends heavily on the training data used for the classifier —
  a model trained on synthetic or very limited samples will not generalize
  well to arbitrary real-world hands, lighting, or backgrounds.
- Only one hand is detected and classified per image (`max_num_hands=1`).

## Extending the Project

- **Add new gestures** — collect labeled samples for the new gesture, extract
  landmarks, and retrain the classifier, then swap in the new `.pkl` file.
- **Improve robustness** — train on a larger, more varied real-world dataset
  rather than synthetic or self-collected samples alone.
- **Trigger real actions** — map predicted labels to actions (e.g. control a
  slideshow, send a signal to another app) with a simple `if/elif` block.
- **Go real-time** — move inference to a local script using
  `cv2.VideoCapture(0)` for continuous, frame-by-frame gesture control.

## Tech Stack

- MediaPipe — hand landmark detection
- scikit-learn — MLPClassifier
- OpenCV (headless) — image handling
- Streamlit — web app framework and deployment
