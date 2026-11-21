# Robofest Weed / Crop Detector

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
git clone https://github.com/<purnendushekharjha>/<Weed-plant detector>.git
cd <Weed-plant detector>
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

## Files

| File | Purpose |
|---|---|
| `weed_crop_detector.py` | Detection pipeline and CLI |
| `robofest_arena.mp4` | Sample arena video |
| `assets/demo.png` | Screenshot used in this README |

## License

MIT – see [LICENSE](LICENSE).
