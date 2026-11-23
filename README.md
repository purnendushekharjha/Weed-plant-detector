# Robofest Vision: Weed/Crop Detector and Line Follower

Two classical computer-vision tools for the Robofest arena (OpenCV + NumPy, no machine learning):

1. `weed_crop_detector.py` : detects weeds and crops
2. `line_follower.py` : follows a black line and reports a steering error

---

## 1. Weed / Crop Detector

A classical computer-vision pipeline (OpenCV + NumPy, no machine learning) that
detects **weeds (red markers)** and **crops (green plants)** in an arena video
or live camera feed, drawing bounding boxes, centroids and confidence scores.

![Demo](assets/demo.png)

## How it works

1. **Gray-world white balance** – reduces colour cast from lighting/camera.
2. **CLAHE on the Lab L-channel** – evens out brightness without touching colour.
3. **Region of interest** – ignores the top 15% of the frame (background).
4. **Colour segmentation**
   - Red (weed): HSV hue wrap-around (0–10 and 170–179).
   - Green (crop): HSV green range, fused with Excess-Green (ExG) and Lab-A evidence.
5. **Morphology** – opening removes noise, closing fills gaps.
6. **Contour analysis** – area, extent, solidity, circularity, centroid.
7. **Confidence score** – weighted mix of colour purity, shape, area (and ExG for crops);
   detections below `MIN_CONFIDENCE` are dropped.

## Setup

```bash
git clone https://github.com/purnendushekharjha/Weed-plant-detector.git
cd Weed-plant-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Run on the included sample video
python weed_crop_detector.py

# Use a webcam
python weed_crop_detector.py --source 0

# Your own video, faster playback, no debug windows
python weed_crop_detector.py --source my_video.mp4 --delay 1 --no-debug
```

Press **Q** to quit.

## Tuning

All thresholds live at the top of `weed_crop_detector.py` (HSV ranges, `MIN_AREA`,
`MIN_CONFIDENCE`, morphology settings). Calibrate the HSV values for your own
camera and lighting. Keep `DEBUG = True` to view the red/green masks while tuning.

---

## 2. Line Follower

Detects the black line on the arena floor and outputs how far it is from the
camera centre, so a robot can steer back to it.

![Line follower](assets/line_follower_demo.png)

1. Crops to a region of interest (bottom 45% of the frame, middle 40% of its width).
2. Thresholds dark pixels in HSV (`V <= 80`).
3. Cleans the mask with morphological opening and closing.
4. Takes the largest contour and finds its centroid.
5. **Raw error** = line centre minus camera centre (negative = line is left, positive = right).
6. **Smooth error** = exponential moving average of the raw error (`ERROR_ALPHA`).
7. Flags a **possible fork** when the detected region is unusually wide.

```bash
python line_follower.py
python line_follower.py --source 0          # webcam
python line_follower.py --delay 1           # faster playback
```

Each frame prints: `raw_error  smooth_error  line_found  fork_detected`. Press **Q** to quit.

---

## Files

| File | Purpose |
|---|---|
| `weed_crop_detector.py` | Weed/crop detection pipeline and CLI |
| `line_follower.py` | Line detection and steering error |
| `robofest_arena.mp4` | Sample arena video |
| `assets/` | Screenshots used in this README |

## License

MIT – see [LICENSE](LICENSE).
