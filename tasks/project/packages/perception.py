import os
from typing import List, Optional

import cv2
import numpy as np
import yaml

from tasks.object_detection.packages.agent import ObjectDetectionAgent
from tasks.project.packages.led_tracker import LedPair, LedTracker
from tasks.project.packages.world_model import (
    LaneObs,
    LeaderObs,
    SignKind,
    SignObs,
    WorldModel,
)
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

_TRUCK_CLS = 1

# Duckietown traffic signs are AprilTags. Map our config's family name to the
# OpenCV aruco predefined-dictionary attribute (all AprilTag families ship in
# cv2.aruco from OpenCV 4.7+, no extra dependency needed).
_APRILTAG_DICTS = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag25h9":  "DICT_APRILTAG_25h9",
    "tag16h5":  "DICT_APRILTAG_16h5",
}

_CONFIG_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml")
)


class Perception:
    def __init__(self, config_path: Optional[str] = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.lane_healthy_min_pixels = cfg.get("lane_healthy_min_pixels", 500)
        self.leader_lower_half_only = cfg.get("leader_lower_half_only", True)
        self.detection_cache_ttl = cfg.get("detection_cache_ttl", 5)
        # Fraction of frame width inside which an LED midpoint and a YOLO
        # bbox center must agree to call the fix "fused".
        self.led_yolo_agree_frac = float(cfg.get("led_yolo_agree_frac", 0.15))

        # AprilTag traffic signs. The lead-bot town uses tag36h11 by default.
        # Map detected tag IDs -> sign kind via config (homemade signs can use
        # any IDs, so read the logged ids and fill these in).
        self.apriltag_family = str(cfg.get("apriltag_family", "tag36h11"))
        self.apriltag_stop_ids = {int(i) for i in (cfg.get("apriltag_stop_ids") or [])}
        self.apriltag_slow_ids = {int(i) for i in (cfg.get("apriltag_slow_ids") or [])}
        # Ignore tags smaller than this (px²) so distant signs don't trigger a
        # stop. 0 = react to any detected tag (presence-based).
        self.apriltag_min_area_px = float(cfg.get("apriltag_min_area_px", 0))

        self.lane = LaneServoingAgent()
        self.detector = ObjectDetectionAgent()
        self.led = LedTracker(cfg)

        self._aruco = None
        self._tag_detector = self._make_tag_detector(self.apriltag_family)
        self._last_tag_ids: List[int] = []

        self._last_dets: List = []
        self._last_dets_age = 999
        self.last_debug_info: dict = {}

    def update(self, frame_bgr: np.ndarray, now: float) -> WorldModel:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        lane_obs = self._lane_obs(frame_rgb)
        dets = self._fresh_detections(frame_rgb)
        led_pair = self.led.update(frame_bgr)
        leader = self._best_leader(dets, led_pair, w, h)
        signs = self._apriltag_signs(gray)

        self.last_debug_info = {
            "led_pair_bbox":   led_pair.bbox     if led_pair else None,
            "led_centroid":    led_pair.midpoint if led_pair else None,
            "led_pair_px":     led_pair.pair_px  if led_pair else None,
            "leader_source":   leader.source     if leader   else None,
            "leader_score":    leader.score      if leader   else None,
            "apriltag_ids":    list(self._last_tag_ids),
        }

        return WorldModel(
            t=now,
            frame_w=w,
            frame_h=h,
            lane=lane_obs,
            leader=leader,
            signs=signs,
            detector_ready=self.detector.model_loaded,
        )

    def _lane_obs(self, frame_rgb: np.ndarray) -> LaneObs:
        left, right = self.lane.compute_commands(frame_rgb)
        debug = self.lane.last_debug_info
        lane_pixels = int(debug.get("total_lane_pixels", 0))
        return LaneObs(
            steering_suggestion=(right - left) / 2.0,
            base_speed_suggestion=(left + right) / 2.0,
            lane_pixels=lane_pixels,
            is_curve=bool(debug.get("is_curve", False)),
            healthy=lane_pixels >= self.lane_healthy_min_pixels,
        )

    def _fresh_detections(self, frame_rgb: np.ndarray) -> List:
        dets = self.detector.detect(frame_rgb)
        # detect() returns None on skip frames — reuse the previous list briefly.
        if dets is None:
            self._last_dets_age += 1
            if self._last_dets_age > self.detection_cache_ttl:
                return []
            return self._last_dets
        self._last_dets = dets
        self._last_dets_age = 0
        return dets

    def _best_truck(self, dets, w: int, h: int):
        candidates = []
        for bbox, score, cls_id in dets:
            if cls_id != _TRUCK_CLS:
                continue
            x1, y1, x2, y2 = bbox
            cy = (y1 + y2) / 2
            if self.leader_lower_half_only and cy < h / 2:
                continue
            area = max(1, (x2 - x1) * (y2 - y1))
            candidates.append((area, bbox, score))
        if not candidates:
            return None
        _, bbox, score = max(candidates, key=lambda c: c[0])
        return bbox, float(score)

    def _best_leader(self, dets, led: Optional[LedPair], w: int, h: int) -> Optional[LeaderObs]:
        truck = self._best_truck(dets, w, h)

        # YOLO + LED -> fused. LED midpoint takes precedence for localisation.
        if truck is not None and led is not None:
            bbox, yscore = truck
            x1, y1, x2, y2 = bbox
            bcx = 0.5 * (x1 + x2)
            agree = abs(bcx - led.midpoint[0]) / max(w, 1) <= self.led_yolo_agree_frac
            cx = led.midpoint[0]
            lateral = float((cx - w / 2) / (w / 2))
            return LeaderObs(
                bbox=led.bbox,
                distance_px=float(y2 - y1),
                lateral=lateral,
                score=float(min(1.0, 0.5 * yscore + 0.5 * led.score) * (1.0 if agree else 0.6)),
                pair_px=led.pair_px,
                source="fused" if agree else "led",
            )

        # LED only.
        if led is not None:
            cx = led.midpoint[0]
            lateral = float((cx - w / 2) / (w / 2))
            return LeaderObs(
                bbox=led.bbox,
                distance_px=float(led.bbox[3] - led.bbox[1]),  # weak fallback for legacy consumers
                lateral=lateral,
                score=float(led.score),
                pair_px=led.pair_px,
                source="led",
            )

        # YOLO only -> previous behaviour.
        if truck is not None:
            bbox, yscore = truck
            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)
            lateral = float((cx - w / 2) / (w / 2))
            return LeaderObs(
                bbox=bbox,
                distance_px=float(y2 - y1),
                lateral=lateral,
                score=yscore,
                pair_px=None,
                source="yolo",
            )

        return None

    def _make_tag_detector(self, family: str):
        """Build an AprilTag detector from cv2.aruco, supporting both the
        modern (>=4.7) ArucoDetector object and the legacy function API.
        Returns None (and warns) if the aruco module is unavailable."""
        try:
            import cv2.aruco as aruco
        except Exception as e:  # pragma: no cover - depends on the cv2 build
            print(f"[project] cv2.aruco unavailable ({e}); AprilTag sign "
                  f"detection is DISABLED. Install an OpenCV build with the "
                  f"aruco module (e.g. opencv-contrib-python).")
            return None

        self._aruco = aruco
        dict_name = _APRILTAG_DICTS.get(str(family).lower(), "DICT_APRILTAG_36h11")
        if not hasattr(aruco, dict_name):
            print(f"[project] AprilTag family '{family}' unsupported by this "
                  f"OpenCV; falling back to tag36h11.")
            dict_name = "DICT_APRILTAG_36h11"
        dict_id = getattr(aruco, dict_name)

        if hasattr(aruco, "getPredefinedDictionary"):
            ad = aruco.getPredefinedDictionary(dict_id)
        else:  # very old API
            ad = aruco.Dictionary_get(dict_id)

        if hasattr(aruco, "ArucoDetector"):  # OpenCV >= 4.7
            params = aruco.DetectorParameters()
            return ("obj", aruco.ArucoDetector(ad, params))
        # Legacy function API.
        params = aruco.DetectorParameters_create()
        return ("fn", (ad, params))

    def _detect_tags(self, gray: np.ndarray):
        """Return (corners, ids) of all AprilTags in the grayscale frame."""
        if self._tag_detector is None:
            return [], []
        mode, det = self._tag_detector
        try:
            if mode == "obj":
                corners, ids, _ = det.detectMarkers(gray)
            else:
                ad, params = det
                corners, ids, _ = self._aruco.detectMarkers(gray, ad, parameters=params)
        except cv2.error:
            return [], []
        if ids is None:
            return [], []
        return corners, [int(i) for i in ids.flatten()]

    def _tag_kind(self, tid: int) -> Optional[SignKind]:
        if tid in self.apriltag_stop_ids:
            return "STOP"
        if tid in self.apriltag_slow_ids:
            return "SLOW"
        return None

    def _apriltag_signs(self, gray: np.ndarray) -> List[SignObs]:
        corners, ids = self._detect_tags(gray)
        self._last_tag_ids = ids  # surfaced in debug so unmapped tags are visible
        out: List[SignObs] = []
        for pts4, tid in zip(corners, ids):
            kind = self._tag_kind(tid)
            if kind is None:
                continue
            pts = np.asarray(pts4).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            if (x2 - x1) * (y2 - y1) < self.apriltag_min_area_px:
                continue
            out.append(SignObs(
                kind=kind,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                score=1.0,
            ))
        return out
