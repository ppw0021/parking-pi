#!/usr/bin/env python3
"""
lens_correction.py
Reusable module to handle fisheye/barrel distortion.
Saves and loads parameters from lens_config.json to ensure consistency across scripts.
"""

import cv2
import numpy as np
import json
import os

CONFIG_FILE = "lens_config.json"

def load_config():
    """Loads lens parameters, returning defaults if the file doesn't exist."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"k1": -0.25, "k2": 0.2}

def save_config(k1, k2):
    """Saves lens parameters to a JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"k1": k1, "k2": k2}, f, indent=4)

def undistort_frame(frame, k1=None, k2=None):
    """
    Applies radial lens distortion correction.
    If k1 and k2 are not provided, it loads them from the config file.
    """
    if k1 is None or k2 is None:
        config = load_config()
        k1 = config.get("k1", -0.25)
        k2 = config.get("k2", 0.2)

    h, w = frame.shape[:2]
    # Approximation of focal length and optical center
    fx, fy = w * 0.8, h * 0.8
    cx, cy = w / 2.0, h / 2.0

    camera_matrix = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
    )
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    return cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
    )