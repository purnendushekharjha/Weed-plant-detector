

import argparse

import cv2
import numpy as np



ROI_TOP_RATIO = 0.55
ROI_BOTTOM_RATIO = 1.00

MIN_CONTOUR_AREA = 300

# Error smoothing
ERROR_ALPHA = 0.30

# Fork detection
FORK_WIDTH_RATIO = 0.45
MIN_FORK_AREA = 1000


# -----------------------------
# State variables
# -----------------------------

previous_smooth_error = 0.0

def hsv_way(frame):

    global previous_smooth_error

    height, width = frame.shape[:2]

    # -------------------------------------------------
    # 1. Select Region Of Interest (ROI)
    # -------------------------------------------------

    top = int(height * ROI_TOP_RATIO)
    bottom = int(height * ROI_BOTTOM_RATIO)

    roi = frame[top:bottom, int(width*0.3):int(width*0.7)]

    # All measurements below are relative to the cropped ROI,
    # not the full frame.
    roi_width = roi.shape[1]

    # -------------------------------------------------
    # 2. Convert to grayscale
    # -------------------------------------------------

    # Black line detection in HSV (low brightness = dark line)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    mask = cv2.inRange(
        hsv,
        lower_black,
        upper_black
    )
    # -------------------------------------------------
    # 5. Morphological filtering
    # -------------------------------------------------

    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -------------------------------------------------
    # 6. Find contours
    # -------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # -------------------------------------------------
    # 7. Remove very small contours
    # -------------------------------------------------

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA
    ]

    debug_view = roi.copy()

    # -------------------------------------------------
    # 8. No line found
    # -------------------------------------------------

    if not valid_contours:

        cv2.putText(
            debug_view,
            "LINE LOST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return None, None, debug_view, False, False

    # -------------------------------------------------
    # 9. Find largest contour
    # -------------------------------------------------

    line_contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(line_contour)

    # -------------------------------------------------
    # 10. Calculate bounding box
    # -------------------------------------------------

    x, y, w, h = cv2.boundingRect(line_contour)

    # -------------------------------------------------
    # 11. Detect possible fork
    #
    # If the detected region becomes unusually wide,
    # it may represent a fork or intersection.
    # -------------------------------------------------

    fork_detected = False

    if (
        w > roi_width * FORK_WIDTH_RATIO
        and contour_area > MIN_FORK_AREA
    ):
        fork_detected = True

    # -------------------------------------------------
    # 12. Calculate contour center
    # -------------------------------------------------

    M = cv2.moments(line_contour)

    if M["m00"] == 0:

        return None, None, debug_view, False, fork_detected

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # -------------------------------------------------
    # 13. Calculate RAW error
    # -------------------------------------------------

    camera_center = roi_width // 2

    raw_error = cx - camera_center

    # -------------------------------------------------
    # 14. Smooth the error
    #
    # EMA:
    #
    # smooth =
    #     alpha * current
    #     +
    #     (1-alpha) * previous
    # -------------------------------------------------

    smooth_error = (
        ERROR_ALPHA * raw_error
        +
        (1 - ERROR_ALPHA) * previous_smooth_error
    )

    previous_smooth_error = smooth_error

    # -------------------------------------------------
    # 16. Draw contour
    # -------------------------------------------------

    cv2.drawContours(
        debug_view,
        [line_contour],
        -1,
        (0, 255, 0),
        3
    )

    # -------------------------------------------------
    # 17. Draw center point
    # -------------------------------------------------

    cv2.circle(
        debug_view,
        (cx, cy),
        8,
        (0, 0, 255),
        -1
    )

    # -------------------------------------------------
    # 18. Draw camera center
    # -------------------------------------------------

    cv2.line(
        debug_view,
        (camera_center, 0),
        (camera_center, debug_view.shape[0]),
        (255, 0, 0),
        2
    )

    # -------------------------------------------------
    # 19. Draw bounding box
    # -------------------------------------------------

    cv2.rectangle(
        debug_view,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        2
    )

    # -------------------------------------------------
    # 20. Display raw and smooth error
    # -------------------------------------------------

    cv2.putText(
        debug_view,
        f"Raw Error: {raw_error:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        debug_view,
        f"Smooth Error: {smooth_error:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # -------------------------------------------------
    # 21. Fork warning
    # -------------------------------------------------

    if fork_detected:

        cv2.putText(
            debug_view,
            "POSSIBLE FORK",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return (
        raw_error,
        smooth_error,
        debug_view,
        True,
        fork_detected
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Robofest line follower: detects a black line and "
                    "reports steering error."
    )
    parser.add_argument("--source", default="robofest_arena.mp4",
                        help="Video file path or camera index (default: robofest_arena.mp4)")
    parser.add_argument("--delay", type=int, default=40,
                        help="Milliseconds between frames (default: 40)")
    return parser.parse_args()


def main():

    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open source: {args.source}")
        return

    print("Robofest Line Follower Started")
    print("Columns: raw_error, smooth_error, line_found, fork_detected")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("End of video / frame capture failed.")
            break

        raw_error, smooth_error, debug_view, found, fork = hsv_way(frame)

        print(raw_error, smooth_error, found, fork)

        cv2.imshow("Line Follower", debug_view)

        if cv2.waitKey(args.delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Line follower stopped.")


if __name__ == "__main__":
    main()
