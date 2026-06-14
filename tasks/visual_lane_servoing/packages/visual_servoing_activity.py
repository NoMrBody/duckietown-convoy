from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 0),  _h.get('yellow_lower_s', 0),  _h.get('yellow_lower_v', 0)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 0),  _h.get('yellow_upper_s', 0), _h.get('yellow_upper_v', 0)])

_white_lower = np.array([_h.get('white_lower_h', 0),   _h.get('white_lower_s', 0), _h.get('white_lower_v', 0)])
_white_upper = np.array([_h.get('white_upper_h', 0), _h.get('white_upper_s', 0), _h.get('white_upper_v', 0)])

_SIGMA = 4.5
_MAG_THRESHOLD = 40.0


def _keep_blobs_touching_edges(color_mask, edge_mask):
    """Keep entire connected components of ``color_mask`` (0/255) that overlap a
    strong edge, dropping flat blobs with no edge anywhere. Unlike pixel-wise
    edge gating this preserves the filled interior of a marking instead of
    leaving only its outline."""
    num, labels = cv2.connectedComponents(color_mask)
    if num <= 1:
        return color_mask
    hit = np.unique(labels[edge_mask & (labels > 0)])
    hit = hit[hit > 0]
    if hit.size == 0:
        return np.zeros_like(color_mask)
    return np.where(np.isin(labels, hit), 255, 0).astype(np.uint8)


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Crop the top (horizon) — only look at the road
    h, w = image.shape[:2]
    crop_top = int(h * 0.4)
    roi = image[crop_top:, :]

    # Blur to reduce noise (kernel derived from _SIGMA)
    blurred = cv2.GaussianBlur(roi, (0, 0), _SIGMA)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Color masks
    yellow_mask = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
    white_mask  = cv2.inRange(hsv, _white_lower, _white_upper)

    # Morphological opening: drop isolated speckle before it reaches the agent,
    # which otherwise floods on the broad white range (bright road / glare).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    white_mask  = cv2.morphologyEx(white_mask,  cv2.MORPH_OPEN, kernel)

    # Edge gating. Pure color can't tell a line from bright road/glare, which
    # share its low-saturation/high-value signature; a real line has a strong
    # intensity gradient, flat road doesn't.
    gray   = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    strong = np.sqrt(sobelx * sobelx + sobely * sobely) > _MAG_THRESHOLD

    # White aliases with bright road/glare, so keep only color pixels that sit on
    # a strong edge with the correct horizontal-gradient sign (right-side edge).
    white_mask = np.where(strong & (sobelx > 0), white_mask, 0).astype(np.uint8)

    # Yellow's hue is distinctive, so don't thin it down to its edge (that left
    # only the border outline, hollowing out thick markings). Keep the WHOLE
    # color blob as long as it touches a strong edge — this fills the interior
    # while still rejecting flat yellow-ish glare that has no edge anywhere.
    yellow_mask = _keep_blobs_touching_edges(yellow_mask, strong)

    # Put masks back into full-size images
    full_yellow = np.zeros((h, w), dtype=np.uint8)
    full_white  = np.zeros((h, w), dtype=np.uint8)
    full_yellow[crop_top:, :] = yellow_mask
    full_white[crop_top:, :]  = white_mask

    return (full_yellow // 255).astype(np.uint8), (full_white // 255).astype(np.uint8)


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower  = np.array(white_lower)
    _white_upper  = np.array(white_upper)


def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]), 'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]), 'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]), 'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]),  'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]),  'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]),  'white_upper_v':  int(_white_upper[2]),
    }