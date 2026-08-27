#!/usr/bin/env python3
"""
focuser.py
Takes a photo every 500ms and overwrites 'focus_test.jpg' 
so you can tune the camera focus over an SSH connection.
Press Ctrl+C in the terminal to stop.
"""

import cv2
import time
import sys

def main(camera_index=0):
    print(f"Starting focus stream on camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        sys.exit(1)

    # Force 1080p so you are focusing at the exact resolution you'll be using
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Running... Open 'focus_test.jpg' in your file explorer.")
    print("Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.imwrite("focus_test.jpg", frame)
            
            # Wait 500ms before taking the next photo
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping focuser...")
    finally:
        cap.release()
        print("Camera released.")

if __name__ == "__main__":
    main()