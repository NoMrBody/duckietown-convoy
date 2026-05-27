from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml'
))

# Sensible fallback bounds for a typical Duckietown environment, used only if
# the YAML config is missing. A silent all-zero default would match nothing
# and stall the bot with no obvious cause.
_FALLBACK = {
    'yellow_lower_h':  15, 'yellow_lower_s':  80, 'yellow_lower_v':  80,
    'yellow_upper_h':  45, 'yellow_upper_s': 255, 'yellow_upper_v': 255,
    'white_lower_h':    0, 'white_lower_s':   0, 'white_lower_v': 180,
    'white_upper_h':  180, 'white_upper_s':  60, 'white_upper_v': 255,
}

try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    print(f"[lane_servoing] HSV config not found at {_HSV_FILE}; using fallback bounds.")
    _h = {}

_h = {**_FALLBACK, **_h}

_yellow_lower = np.array([_h['yellow_lower_h'], _h['yellow_lower_s'], _h['yellow_lower_v']])
_yellow_upper = np.array([_h['yellow_upper_h'], _h['yellow_upper_s'], _h['yellow_upper_v']])

_white_lower = np.array([_h['white_lower_h'], _h['white_lower_s'], _h['white_lower_v']])
_white_upper = np.array([_h['white_upper_h'], _h['white_upper_s'], _h['white_upper_v']])

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects the left (dashed-yellow) and right (solid-white) lane markings.

    Args:
        image: BGR image from the Duckiebot's camera (H x W x 3, uint8)

    Returns:
        mask_left:  binary mask of the left (yellow) lane marking
        mask_right: binary mask of the right (white) lane marking
    """
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_ground = np.zeros((h, w), dtype=np.uint8)
    mask_ground[h // 3:, :] = 1

    # Widened split: allow yellow up to 60% from the left and white from 40%,
    # so a sharp curve that pushes a line briefly across the centerline does
    # not erase it.
    left_cut  = int(w * 0.60)
    right_cut = int(w * 0.40)
    mask_left_half  = np.zeros((h, w), dtype=np.uint8)
    mask_right_half = np.zeros((h, w), dtype=np.uint8)
    mask_left_half[:, :left_cut]   = 1
    mask_right_half[:, right_cut:] = 1

    mask_yellow = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
    mask_white  = cv2.inRange(hsv, _white_lower,  _white_upper)

    # Open to drop isolated noise specks before they pull centroid estimates off the line.
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_OPEN, _MORPH_KERNEL)

    mask_yellow = (mask_yellow > 0).astype(np.uint8)
    mask_white  = (mask_white  > 0).astype(np.uint8)

    mask_left  = mask_ground * mask_left_half  * mask_yellow
    mask_right = mask_ground * mask_right_half * mask_white

    return mask_left, mask_right

def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper, persist: bool = True):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower  = np.array(white_lower)
    _white_upper  = np.array(white_upper)
    if persist:
        try:
            with open(_HSV_FILE, 'w') as f:
                yaml.safe_dump(get_hsv_bounds(), f, sort_keys=True)
        except OSError as e:
            print(f"[lane_servoing] could not persist HSV bounds to {_HSV_FILE}: {e}")

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
    }
