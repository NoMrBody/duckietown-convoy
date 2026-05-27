import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

_HSV_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "lead_led_hsv_config.yaml"
))

_FALLBACK_HSV = {
    "red_low_h_min":  0,  "red_low_h_max":  10,
    "red_high_h_min": 160, "red_high_h_max": 180,
    "s_min": 120, "s_max": 255,
    "v_min": 120, "v_max": 255,
}

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class LedPair:
    """Result of a successful pair detection on one frame."""
    midpoint: Tuple[float, float]
    pair_px: float
    bbox: Tuple[int, int, int, int]   # bbox spanning both blobs
    avg_area: float
    score: float


def _load_hsv() -> dict:
    try:
        with open(_HSV_FILE) as f:
            return {**_FALLBACK_HSV, **(yaml.safe_load(f) or {})}
    except FileNotFoundError:
        print(f"[led_tracker] HSV config not found at {_HSV_FILE}; using fallback bounds.")
        return dict(_FALLBACK_HSV)


def red_mask(frame_bgr: np.ndarray, hsv: dict) -> np.ndarray:
    """Two-range red mask (handles hue wrap at 0)."""
    hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(
        hsv_img,
        np.array([hsv["red_low_h_min"],  hsv["s_min"], hsv["v_min"]]),
        np.array([hsv["red_low_h_max"],  hsv["s_max"], hsv["v_max"]]),
    )
    high = cv2.inRange(
        hsv_img,
        np.array([hsv["red_high_h_min"], hsv["s_min"], hsv["v_min"]]),
        np.array([hsv["red_high_h_max"], hsv["s_max"], hsv["v_max"]]),
    )
    mask = cv2.bitwise_or(low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return mask


def _candidate_blobs(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> List[Tuple[float, float, float, Tuple[int, int, int, int]]]:
    """Return list of (cx, cy, area, bbox) for valid red blobs."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):  # skip label 0 (background)
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if bw == 0 or bh == 0:
            continue
        aspect = bw / float(bh)
        if aspect < 0.4 or aspect > 2.5:
            continue
        cx, cy = centroids[i]
        out.append((float(cx), float(cy), float(area), (int(x), int(y), int(x + bw), int(y + bh))))
    return out


def _score_pair(
    a, b,
    y_tol_px: float,
    min_sep_px: float,
    max_sep_px: float,
) -> float:
    """Higher is better. Zero means reject."""
    (ax, ay, aa, _), (bx, by, ba, _) = a, b
    dy = abs(ay - by)
    if dy > y_tol_px:
        return 0.0
    dx = abs(ax - bx)
    if dx < min_sep_px or dx > max_sep_px:
        return 0.0
    ratio = min(aa, ba) / max(aa, ba)
    if ratio < 0.5:
        return 0.0
    # Reward similar y, similar area, comfortable horizontal separation.
    s_y    = 1.0 - dy / max(y_tol_px, 1.0)
    s_area = ratio
    s_sep  = 1.0 - abs(dx - 0.5 * (min_sep_px + max_sep_px)) / max(0.5 * (max_sep_px - min_sep_px), 1.0)
    s_sep  = max(0.0, s_sep)
    return float(0.4 * s_y + 0.4 * s_area + 0.2 * s_sep)


class LedTracker:
    """Detects a pair of steady-on red LEDs (the lead bot's rear lights) and tracks them across frames."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.min_area         = int(cfg.get("led_min_area_px",         4))
        self.max_area         = int(cfg.get("led_max_area_px",         400))
        self.y_tol_px         = float(cfg.get("led_y_tol_px",          15))
        self.min_sep_px       = float(cfg.get("led_min_sep_px",        10))
        self.max_sep_px       = float(cfg.get("led_max_sep_px",        200))
        self.roi_top_frac     = float(cfg.get("led_roi_top_frac",      0.30))
        self.roi_bot_frac     = float(cfg.get("led_roi_bot_frac",      0.95))
        self.track_radius_px  = float(cfg.get("led_track_radius_px",   60))
        self.lost_grace       = int(cfg.get("led_lost_grace_frames",   5))
        self.pair_px_max_jump = float(cfg.get("led_pair_px_max_jump",  25))

        self._hsv = _load_hsv()
        self._last: Optional[LedPair] = None
        self._missed = 0

    def reload_hsv(self) -> None:
        self._hsv = _load_hsv()

    def _roi_bounds(self, h: int) -> Tuple[int, int]:
        y_top = max(0, int(self.roi_top_frac * h))
        y_bot = min(h, int(self.roi_bot_frac * h))
        return y_top, y_bot

    def _pick_best_pair(
        self,
        blobs: List[Tuple[float, float, float, Tuple[int, int, int, int]]],
        bias_to_last: bool,
    ) -> Optional[LedPair]:
        if len(blobs) < 2:
            return None

        best: Optional[LedPair] = None
        best_combined = -1.0
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                geom = _score_pair(
                    blobs[i], blobs[j],
                    self.y_tol_px, self.min_sep_px, self.max_sep_px,
                )
                if geom == 0.0:
                    continue
                (ax, ay, aa, abox), (bx, by, ba, bbox) = blobs[i], blobs[j]
                mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                pair_px = float(np.hypot(ax - bx, ay - by))
                merged_bbox = (
                    min(abox[0], bbox[0]), min(abox[1], bbox[1]),
                    max(abox[2], bbox[2]), max(abox[3], bbox[3]),
                )

                continuity = 1.0
                if bias_to_last and self._last is not None:
                    lx, ly = self._last.midpoint
                    dist = float(np.hypot(mx - lx, my - ly))
                    cont_xy = max(0.0, 1.0 - dist / max(self.track_radius_px * 1.5, 1.0))
                    pair_jump = abs(pair_px - self._last.pair_px)
                    if pair_jump > self.pair_px_max_jump:
                        continue
                    cont_sz = max(0.0, 1.0 - pair_jump / max(self.pair_px_max_jump, 1.0))
                    continuity = 0.5 * cont_xy + 0.5 * cont_sz

                combined = geom * (0.5 + 0.5 * continuity)
                if combined > best_combined:
                    best_combined = combined
                    best = LedPair(
                        midpoint=(mx, my),
                        pair_px=pair_px,
                        bbox=merged_bbox,
                        avg_area=0.5 * (aa + ba),
                        score=float(min(1.0, combined)),
                    )
        return best

    def update(self, frame_bgr: np.ndarray) -> Optional[LedPair]:
        h, w = frame_bgr.shape[:2]
        y_top, y_bot = self._roi_bounds(h)

        mask_full = red_mask(frame_bgr, self._hsv)
        # Restrict to the vertical ROI by zeroing above/below.
        mask = np.zeros_like(mask_full)
        mask[y_top:y_bot, :] = mask_full[y_top:y_bot, :]

        result: Optional[LedPair] = None

        if self._last is not None:
            lx, ly = self._last.midpoint
            rx1 = max(0, int(lx - self.track_radius_px))
            rx2 = min(w, int(lx + self.track_radius_px))
            ry1 = max(y_top, int(ly - self.track_radius_px))
            ry2 = min(y_bot, int(ly + self.track_radius_px))
            if rx2 > rx1 and ry2 > ry1:
                roi_mask = np.zeros_like(mask)
                roi_mask[ry1:ry2, rx1:rx2] = mask[ry1:ry2, rx1:rx2]
                roi_blobs = _candidate_blobs(roi_mask, self.min_area, self.max_area)
                result = self._pick_best_pair(roi_blobs, bias_to_last=True)

        if result is None:
            full_blobs = _candidate_blobs(mask, self.min_area, self.max_area)
            result = self._pick_best_pair(full_blobs, bias_to_last=self._last is not None)

        if result is not None:
            self._last = result
            self._missed = 0
            return result

        # Grace: reuse the last estimate briefly so single-frame misses don't drop the lock.
        if self._last is not None:
            self._missed += 1
            if self._missed <= self.lost_grace:
                return self._last
            self._last = None
            self._missed = 0
        return None
