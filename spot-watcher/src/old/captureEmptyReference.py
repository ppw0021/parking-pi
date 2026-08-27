#!/usr/bin/env python3
import time
import cv2


def main():
    print(
        "📷 Starting 1080p webcam capture — saving continuously as emptyReferenceFocus.jpg (1 photo/sec)"
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("❌ Could not access webcam")

    # Force MJPEG mode & 1080p resolution for Bonelk webcam
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Sensor warmup delay
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Failed to capture frame, retrying...")
            time.sleep(1)
            continue

        cv2.imwrite("emptyReferenceFocus.jpg", frame)
        print("💾 Updated emptyReferenceFocus.jpg (1080p)")
        time.sleep(1)


if __name__ == "__main__":
    main()