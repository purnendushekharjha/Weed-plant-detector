import argparse

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ------------------------------------------------------------
# ROI
# ------------------------------------------------------------

ROI_TOP_RATIO = 0.15
ROI_BOTTOM_RATIO = 1.00

# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

MIN_AREA = 120

# Ignore objects touching the very edge of ROI if desired
IGNORE_EDGE_OBJECTS = False
EDGE_MARGIN = 3

# ------------------------------------------------------------
# Morphology
# ------------------------------------------------------------

MORPH_KERNEL = 5

OPEN_ITERATIONS = 1
CLOSE_ITERATIONS = 2

# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

MIN_CONFIDENCE = 0.55

# ------------------------------------------------------------
# Debug
# ------------------------------------------------------------

DEBUG = True


# ============================================================
# HSV PARAMETERS
# ============================================================
#
# OpenCV HSV:
#
# H = 0 ... 179
# S = 0 ... 255
# V = 0 ... 255
#
# RED wraps around the HSV boundary.
#
# These values are STARTING VALUES.
# Calibrate them using your actual camera.
# ============================================================


# RED
RED_H1_LOW = 0
RED_H1_HIGH = 10

RED_H2_LOW = 170
RED_H2_HIGH = 179

RED_S_LOW = 70
RED_V_LOW = 40


# GREEN
GREEN_H_LOW = 30
GREEN_H_HIGH = 95

GREEN_S_LOW = 45
GREEN_V_LOW = 35


# ============================================================
# EXG PARAMETERS
# ============================================================

# ExG is used as supporting evidence for vegetation.
#
# Normalized ExG:
#
# ExG = 2g - r - b
#
# where r,g,b are normalized RGB channels.
#
# Range approximately:
# -2 ... +2
#
# We convert it to 0...255.

EXG_THRESHOLD = 25


# ============================================================
# LAB PARAMETERS
# ============================================================

# OpenCV Lab:
#
# L = 0...255
# A = 0...255
# B = 0...255
#
# Green generally moves toward lower A.
#
# This is only supporting evidence.

LAB_GREEN_A_MAX = 145



def gray_world_white_balance(frame):
    """
    Simple gray-world white balance.

    Attempts to reduce color cast caused by:
        sunlight
        artificial lighting
        camera color bias
    """

    img = frame.astype(np.float32)

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg = (avg_b + avg_g + avg_r) / 3.0

    # Avoid division problems
    avg_b = max(avg_b, 1.0)
    avg_g = max(avg_g, 1.0)
    avg_r = max(avg_r, 1.0)

    img[:, :, 0] *= avg / avg_b
    img[:, :, 1] *= avg / avg_g
    img[:, :, 2] *= avg / avg_r

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)




def normalize_lighting(frame):
    """
    LAB + CLAHE.

    CLAHE is applied only to the L channel,
    so we avoid directly changing the color channels.
    """

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    L = clahe.apply(L)

    lab = cv2.merge([
        L,
        A,
        B
    ])

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )


def calculate_exg(frame):
    """
    Calculate normalized Excess Green Index.

    ExG = 2g - r - b

    Returns:
        exg_visual : 0...255 image
        exg_raw    : floating-point ExG
    """

    img = frame.astype(np.float32)

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    total = R + G + B + 1e-6

    r = R / total
    g = G / total
    b = B / total

    exg = (
        2.0 * g
        - r
        - b
    )

    # Convert approximately -1...1 to 0...255
    exg_visual = (
        (exg + 1.0) * 127.5
    )

    exg_visual = np.clip(
        exg_visual,
        0,
        255
    ).astype(np.uint8)

    return exg_visual, exg



def create_hsv_masks(hsv):

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    red_hue = (
        ((H >= RED_H1_LOW) &
         (H <= RED_H1_HIGH))
        |
        ((H >= RED_H2_LOW) &
         (H <= RED_H2_HIGH))
    )

    red_mask = (
        red_hue
        &
        (S >= RED_S_LOW)
        &
        (V >= RED_V_LOW)
    )

    red_mask = (
        red_mask.astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------

    green_mask = (
        (H >= GREEN_H_LOW)
        &
        (H <= GREEN_H_HIGH)
        &
        (S >= GREEN_S_LOW)
        &
        (V >= GREEN_V_LOW)
    )

    green_mask = (
        green_mask.astype(np.uint8)
        * 255
    )

    return red_mask, green_mask






def create_lab_green_evidence(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    A = lab[:, :, 1]

    evidence = (
        A <= LAB_GREEN_A_MAX
    )

    return (
        evidence.astype(np.uint8)
        * 255
    )




# ============================================================
# CREATE FINAL COLOR MASKS
# ============================================================

def create_masks(frame):

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    red_hsv, green_hsv = (
        create_hsv_masks(hsv)
    )

    # --------------------------------------------------------
    # ExG
    # --------------------------------------------------------

    exg_visual, exg_raw = (
        calculate_exg(frame)
    )

    exg_green = (
        exg_raw > (EXG_THRESHOLD / 127.5 - 1)
    )

    exg_green = (
        exg_green.astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Lab
    # --------------------------------------------------------

    lab_green = (
        create_lab_green_evidence(
            frame
        )
    )

    # ========================================================
    # GREEN MASK FUSION
    # ========================================================
    #
    # HSV is the primary detector.
    #
    # ExG and Lab provide supporting evidence.
    #
    # We DO NOT require:
    #
    # HSV AND ExG AND Lab
    #
    # because that would be too restrictive.
    #
    # Instead:
    #
    # HSV GREEN
    #       AND
    # (ExG OR Lab)
    #
    # ========================================================

    vegetation_evidence = cv2.bitwise_or(
        exg_green,
        lab_green
    )

    green_mask = cv2.bitwise_and(
        green_hsv,
        vegetation_evidence
    )

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------
    #
    # Red is primarily HSV based.
    #
    # ExG is intentionally NOT used to reject red.
    # --------------------------------------------------------

    red_mask = red_hsv

    # --------------------------------------------------------
    # Prevent overlap
    # --------------------------------------------------------

    overlap = cv2.bitwise_and(
        red_mask,
        green_mask
    )

    green_mask[
        overlap > 0
    ] = 0

    red_mask[
        overlap > 0
    ] = 0

    return (
        red_mask,
        green_mask,
        exg_visual,
        hsv
    )




def clean_mask(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL,
            MORPH_KERNEL
        )
    )

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------
    #
    # Removes tiny isolated noise.
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=OPEN_ITERATIONS
    )

    # --------------------------------------------------------
    # CLOSING
    # --------------------------------------------------------
    #
    # Connects nearby pixels and fills
    # small gaps.
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=CLOSE_ITERATIONS
    )

    return mask


def get_contour_features(contour):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    M = cv2.moments(
        contour
    )

    if M["m00"] != 0:

        cx = int(
            M["m10"] / M["m00"]
        )

        cy = int(
            M["m01"] / M["m00"]
        )

    else:

        cx = x + w // 2
        cy = y + h // 2

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = (
        w / float(h)
        if h > 0
        else 0
    )

    # --------------------------------------------------------
    # Extent
    # --------------------------------------------------------

    bounding_area = w * h

    extent = (
        area / bounding_area
        if bounding_area > 0
        else 0
    )

    # --------------------------------------------------------
    # Solidity
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    solidity = (
        area / hull_area
        if hull_area > 0
        else 0
    )

    # --------------------------------------------------------
    # Circularity
    # --------------------------------------------------------

    circularity = (
        4.0 * np.pi * area
        / (perimeter * perimeter)
        if perimeter > 0
        else 0
    )

    return {

        "contour": contour,

        "area": area,

        "bbox": (
            x,
            y,
            w,
            h
        ),

        "centroid": (
            cx,
            cy
        ),

        "aspect_ratio":
            aspect_ratio,

        "extent":
            extent,

        "solidity":
            solidity,

        "circularity":
            circularity
    }

def detect_objects(
    mask,
    label,
    frame_shape
):

    height, width = (
        frame_shape[:2]
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for contour in contours:

        features = (
            get_contour_features(
                contour
            )
        )

        area = features["area"]

        # ----------------------------------------------------
        # Area filtering
        # ----------------------------------------------------

        if area < MIN_AREA:
            continue

        x, y, w, h = (
            features["bbox"]
        )

        # ----------------------------------------------------
        # Edge filtering
        # ----------------------------------------------------

        if IGNORE_EDGE_OBJECTS:

            if (
                x <= EDGE_MARGIN
                or y <= EDGE_MARGIN
                or x + w >= width - EDGE_MARGIN
                or y + h >= height - EDGE_MARGIN
            ):
                continue

        # ----------------------------------------------------
        # Add label
        # ----------------------------------------------------

        features["label"] = label

        objects.append(
            features
        )

    return objects

def calculate_color_confidence(
    frame,
    obj,
    label
):

    x, y, w, h = (
        obj["bbox"]
    )

    if w <= 0 or h <= 0:
        return 0.0

    roi = frame[
        y:y+h,
        x:x+w
    ]

    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Only evaluate sufficiently
    # saturated pixels.

    valid = (
        S >= 35
    )

    if np.count_nonzero(
        valid
    ) == 0:

        return 0.0

    H_valid = H[valid]

    # ========================================================
    # RED
    # ========================================================

    if label == "WEED":

        red_pixels = (
            ((H_valid >= 0) &
             (H_valid <= 10))
            |
            ((H_valid >= 170) &
             (H_valid <= 179))
        )

        confidence = (
            np.count_nonzero(
                red_pixels
            )
            /
            len(H_valid)
        )

    # ========================================================
    # GREEN
    # ========================================================

    else:

        green_pixels = (
            (H_valid >= GREEN_H_LOW)
            &
            (H_valid <= GREEN_H_HIGH)
        )

        confidence = (
            np.count_nonzero(
                green_pixels
            )
            /
            len(H_valid)
        )

    return float(
        confidence
    )

def calculate_exg_confidence(
    frame,
    obj
):

    x, y, w, h = (
        obj["bbox"]
    )

    roi = frame[
        y:y+h,
        x:x+w
    ]

    if roi.size == 0:
        return 0.0

    _, exg = (
        calculate_exg(roi)
    )

    # Positive ExG indicates
    # green vegetation evidence.

    confidence = np.mean(
        exg > 0
    )

    return float(
        confidence
    )

def calculate_object_confidence(
    frame,
    obj
):

    label = obj["label"]

    color_score = (
        calculate_color_confidence(
            frame,
            obj,
            label
        )
    )

    # --------------------------------------------------------
    # Shape evidence
    # --------------------------------------------------------

    extent = obj["extent"]
    solidity = obj["solidity"]

    shape_score = (
        0.5 * np.clip(
            extent / 0.5,
            0,
            1
        )
        +
        0.5 * np.clip(
            solidity,
            0,
            1
        )
    )

    # --------------------------------------------------------
    # Area evidence
    # --------------------------------------------------------

    area = obj["area"]

    area_score = np.clip(
        area / 1500.0,
        0,
        1
    )

    # --------------------------------------------------------
    # ExG
    # --------------------------------------------------------

    if label == "CROP":

        exg_score = (
            calculate_exg_confidence(
                frame,
                obj
            )
        )

    else:

        # ExG is not useful as positive
        # evidence for red weed.

        exg_score = 0.0

    # ========================================================
    # FINAL CONFIDENCE
    # ========================================================

    if label == "CROP":

        confidence = (
            0.55 * color_score
            +
            0.20 * exg_score
            +
            0.15 * shape_score
            +
            0.10 * area_score
        )

    else:

        confidence = (
            0.70 * color_score
            +
            0.20 * shape_score
            +
            0.10 * area_score
        )

    return float(
        np.clip(
            confidence,
            0,
            1
        )
    )

def classify_objects(
    frame,
    objects
):

    final_objects = []

    for obj in objects:

        confidence = (
            calculate_object_confidence(
                frame,
                obj
            )
        )

        obj["confidence"] = confidence

        if confidence >= MIN_CONFIDENCE:

            final_objects.append(
                obj
            )

    return final_objects

def draw_results(
    frame,
    objects
):

    output = frame.copy()

    for obj in objects:

        x, y, w, h = (
            obj["bbox"]
        )

        cx, cy = (
            obj["centroid"]
        )

        label = obj["label"]

        confidence = (
            obj["confidence"]
        )

        # ----------------------------------------------------
        # Red = Weed
        # Green = Crop
        # ----------------------------------------------------

        if label == "WEED":

            color = (
                0,
                0,
                255
            )

        else:

            color = (
                0,
                255,
                0
            )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        cv2.circle(
            output,
            (cx, cy),
            5,
            color,
            -1
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text = (
            f"{label} "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            output,
            text,
            (
                x,
                max(y - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        area_text = (
            f"A:{obj['area']:.0f}"
        )

        cv2.putText(
            output,
            area_text,
            (
                x,
                min(
                    y + h + 20,
                    output.shape[0] - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return output

def draw_roi(
    frame,
    top
):

    output = frame.copy()

    cv2.line(
        output,
        (0, top),
        (output.shape[1], top),
        (255, 255, 0),
        2
    )

    return output


# ============================================================
# PIPELINE
# ============================================================

def process_frame(frame):
    """
    Run the full detection pipeline on one BGR frame.

    Returns:
        display     : annotated full frame
        weed_objects, crop_objects : detections (after confidence filter)
        debug       : dict with red_mask, green_mask, exg images
    """

    # 1. White balance
    balanced = gray_world_white_balance(frame)

    # 2. Lighting normalization
    corrected = normalize_lighting(balanced)

    # 3. ROI
    height, _ = corrected.shape[:2]
    roi_top = int(height * ROI_TOP_RATIO)
    roi_bottom = int(height * ROI_BOTTOM_RATIO)
    roi = corrected[roi_top:roi_bottom, :]

    # 4. Color segmentation
    red_mask, green_mask, exg, _ = create_masks(roi)

    # 5. Morphology
    red_mask = clean_mask(red_mask)
    green_mask = clean_mask(green_mask)

    # 6. Contours
    weed_objects = detect_objects(red_mask, "WEED", roi.shape)
    crop_objects = detect_objects(green_mask, "CROP", roi.shape)

    # 7. Confidence filtering
    weed_objects = classify_objects(roi, weed_objects)
    crop_objects = classify_objects(roi, crop_objects)

    # 8. Draw
    result = draw_results(roi, weed_objects + crop_objects)

    display = corrected.copy()
    display[roi_top:roi_bottom, :] = result
    display = draw_roi(display, roi_top)

    cv2.putText(display, f"Weeds: {len(weed_objects)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display, f"Crops: {len(crop_objects)}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    debug = {"red_mask": red_mask, "green_mask": green_mask, "exg": exg}

    return display, weed_objects, crop_objects, debug


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Robofest weed/crop detector (classical computer vision)."
    )

    parser.add_argument(
        "--source",
        default="robofest_arena.mp4",
        help="Video file path, or a camera index such as 0 "
             "(default: robofest_arena.mp4)",
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=40,
        help="Milliseconds between frames (default: 40, ~25 fps)",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Hide the mask / ExG debug windows",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open source: {args.source}")
        return

    # Only meaningful for live cameras; ignored for video files.
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    show_debug = DEBUG and not args.no_debug

    print("Robofest Weed/Crop Detector Started")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("End of video / frame capture failed.")
            break

        display, _, _, debug = process_frame(frame)

        cv2.imshow("ROBOfest Detector", display)

        if show_debug:
            cv2.imshow("RED - WEED MASK", debug["red_mask"])
            cv2.imshow("GREEN - CROP MASK", debug["green_mask"])
            cv2.imshow("ExG", debug["exg"])

        if cv2.waitKey(args.delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Detector stopped.")


if __name__ == "__main__":
    main()
