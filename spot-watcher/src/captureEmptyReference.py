#!/usr/bin/env python3
import cv2
import time

def main():
    print("📷 Starting webcam capture — saving repeatedly as empty_reference.jpg (1 photo/sec)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("Could not access webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image, retrying...")
            time.sleep(1)
            continue
        
        cv2.imwrite("empty_reference.jpg", frame)
        print("Updated empty_reference.jpg")
        time.sleep(1)

if __name__ == "__main__":
    main()
