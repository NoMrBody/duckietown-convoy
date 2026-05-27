from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SignKind = str
Bbox = Tuple[int, int, int, int]
LeaderSource = str  # "yolo" | "led" | "fused"


@dataclass
class LeaderObs:
    bbox: Bbox
    distance_px: float
    lateral: float
    score: float
    pair_px: Optional[float] = None
    source: LeaderSource = "yolo"


@dataclass
class SignObs:
    kind: SignKind
    bbox: Bbox
    score: float


@dataclass
class LaneObs:
    steering_suggestion: float
    base_speed_suggestion: float
    lane_pixels: int
    is_curve: bool
    healthy: bool


@dataclass
class WorldModel:
    t: float
    frame_w: int
    frame_h: int
    lane: LaneObs
    leader: Optional[LeaderObs] = None
    signs: List[SignObs] = field(default_factory=list)
    detector_ready: bool = True
