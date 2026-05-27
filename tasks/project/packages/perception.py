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
_SIGN_CLS = 2

_PAYLOAD_TO_KIND = {
    "STOP": "STOP",
    "SLOW": "SLOW",
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
        self.qr_search_factor = cfg.get("qr_search_factor", 1.5)
        self.detection_cache_ttl = cfg.get("detection_cache_ttl", 5)
        # Fraction of frame width inside which an LED midpoint and a YOLO
        # bbox center must agree to call the fix "fused".
        self.led_yolo_agree_frac = float(cfg.get("led_yolo_agree_frac", 0.15))

        self.lane = LaneServoingAgent()
        self.detector = ObjectDetectionAgent()
        self._qr = cv2.QRCodeDetector()
        self.led = LedTracker(cfg)

        self._last_dets: List = []
        self._last_dets_age = 999
        self.last_debug_info: dict = {}

    def update(self, frame_bgr: np.ndarray, now: float) -> WorldModel:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        lane_obs = self._lane_obs(frame_rgb)
        dets = self._fresh_detections(frame_rgb)
        led_pair = self.led.update(frame_bgr)
        leader = self._best_leader(dets, led_pair, w, h)
        signs = self._decoded_signs(frame_rgb, dets)

        self.last_debug_info = {
            "led_pair_bbox":   led_pair.bbox     if led_pair else None,
            "led_centroid":    led_pair.midpoint if led_pair else None,
            "led_pair_px":     led_pair.pair_px  if led_pair else None,
            "leader_source":   leader.source     if leader   else None,
            "leader_score":    leader.score      if leader   else None,
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

    def _decoded_signs(self, frame_rgb: np.ndarray, dets) -> List[SignObs]:
        out: List[SignObs] = []
        h, w = frame_rgb.shape[:2]
        for bbox, score, cls_id in dets:
            if cls_id != _SIGN_CLS:
                continue
            kind = self._decode_qr_below(frame_rgb, bbox, w, h)
            if kind is None:
                continue
            out.append(SignObs(kind=kind, bbox=bbox, score=float(score)))
        return out

    def _decode_qr_below(self, frame_rgb: np.ndarray, bbox, w: int, h: int) -> Optional[SignKind]:
        x1, y1, x2, y2 = bbox
        box_h = max(1, y2 - y1)
        cy1 = y2
        cy2 = min(h, y2 + int(self.qr_search_factor * box_h))
        cx1 = max(0, x1)
        cx2 = min(w, x2)
        if cy2 <= cy1 or cx2 <= cx1:
            return None
        crop = frame_rgb[cy1:cy2, cx1:cx2]
        try:
            payload, _, _ = self._qr.detectAndDecode(crop)
        except cv2.error:
            return None
        if not payload:
            return None
        return _PAYLOAD_TO_KIND.get(payload.strip().upper())
