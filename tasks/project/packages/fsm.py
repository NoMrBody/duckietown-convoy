from dataclasses import dataclass, field
from typing import Dict, Tuple

from tasks.project.packages.world_model import WorldModel

LedColor = Tuple[float, float, float]

STATE_STOP = "STOP_AT_SIGN"
STATE_SLOW = "SLOW_ZONE"
STATE_FOLLOW = "FOLLOW"
STATE_LANE = "LANE_FOLLOW"
STATE_HOLD = "HOLD"


@dataclass
class Decision:
    state_name: str
    base_speed: float
    steering: float
    leds: Dict[int, LedColor] = field(default_factory=dict)


class ConvoyFSM:
    def __init__(self, cfg: dict):
        self.cruise_speed = float(cfg.get("cruise_speed", 0.3))
        self.slow_factor = float(cfg.get("slow_factor", 0.5))
        self.stop_duration = float(cfg.get("stop_duration", 1.0))
        self.slow_duration = float(cfg.get("slow_duration", 1.5))
        self.stop_cooldown = float(cfg.get("stop_cooldown", 3.0))
        self.slow_cooldown = float(cfg.get("slow_cooldown", 3.0))

        self.safe_distance_px = float(cfg.get("safe_distance_px", 60))
        self.stop_distance_px = float(cfg.get("stop_distance_px", 180))
        # Range proxy from inter-LED pixel distance (used when source is led/fused).
        # Small pair_px = far away; large = close.
        self.led_safe_pair_px = float(cfg.get("led_safe_pair_px", 20))
        self.led_stop_pair_px = float(cfg.get("led_stop_pair_px", 110))
        # Minimum LeaderObs score required to enter FOLLOW from an LED-only fix.
        self.led_min_score = float(cfg.get("led_min_score", 0.4))
        self.leader_lost_grace_s = float(cfg.get("leader_lost_grace_s", 0.25))
        self.leader_p_gain = float(cfg.get("leader_p_gain", 0.6))
        self.max_steer = float(cfg.get("max_steer", 0.4))
        # LED indices to colour red as a "leader-locked" rear signal for any
        # follower behind us. Duckiebot layouts vary; configurable.
        self.rear_led_indices = list(cfg.get("rear_led_indices", [3, 4]))

        self._stop_until = -1.0
        self._slow_until = -1.0
        self._stop_cooldown_until = -1.0
        self._slow_cooldown_until = -1.0
        self._last_leader_t = -1.0

    def step(self, wm: WorldModel) -> Decision:
        self._ingest_signs(wm)
        if self._leader_is_confident(wm.leader):
            self._last_leader_t = wm.t

        if wm.t < self._stop_until:
            return self._make(STATE_STOP, 0.0, 0.0, _RED)

        if wm.t < self._slow_until:
            steering = self._pick_steering(wm)
            return self._make(STATE_SLOW, self.cruise_speed * self.slow_factor, steering, _YELLOW)

        if self._leader_present(wm):
            if wm.leader is not None:
                speed = self._follow_speed(wm.leader)
                steering = self._pick_steering(wm)
            else:
                # Grace window after a brief drop: keep moving forward via the
                # lane signal while waiting to reacquire the leader.
                speed = self.cruise_speed
                steering = wm.lane.steering_suggestion
            return self._make_follow(speed, steering)

        if wm.lane.healthy:
            return self._make(STATE_LANE, self.cruise_speed, wm.lane.steering_suggestion, _BLUE)

        return self._make(STATE_HOLD, 0.0, 0.0, _OFF)

    def _leader_is_confident(self, leader) -> bool:
        if leader is None:
            return False
        if leader.source == "led" and leader.score < self.led_min_score:
            return False
        return True

    def _ingest_signs(self, wm: WorldModel) -> None:
        kinds = {s.kind for s in wm.signs}
        if "STOP" in kinds and wm.t >= self._stop_cooldown_until:
            self._stop_until = wm.t + self.stop_duration
            self._stop_cooldown_until = wm.t + self.stop_duration + self.stop_cooldown
        if "SLOW" in kinds and wm.t >= self._slow_cooldown_until:
            self._slow_until = wm.t + self.slow_duration
            self._slow_cooldown_until = wm.t + self.slow_duration + self.slow_cooldown

    def _leader_present(self, wm: WorldModel) -> bool:
        if self._leader_is_confident(wm.leader):
            return True
        return (wm.t - self._last_leader_t) < self.leader_lost_grace_s

    def _follow_speed(self, leader) -> float:
        if leader.pair_px is not None:
            # Pair_px grows as the lead bot approaches: small = far, large = close.
            safe = self.led_safe_pair_px
            stop = self.led_stop_pair_px
            if stop <= safe:
                return self.cruise_speed
            close = (leader.pair_px - safe) / (stop - safe)
        else:
            safe = self.safe_distance_px
            stop = self.stop_distance_px
            if stop <= safe:
                return self.cruise_speed
            close = (leader.distance_px - safe) / (stop - safe)
        close = max(0.0, min(1.0, close))
        return self.cruise_speed * (1.0 - close)

    def _pick_steering(self, wm: WorldModel) -> float:
        if wm.leader is not None and not wm.lane.healthy:
            # leader.lateral is +ve when the leader is on the RIGHT; the wheel mix
            # (left=base-steer, right=base+steer) needs +ve steer to turn RIGHT, so
            # negate to steer TOWARD the leader (matches the lane convention).
            steer = -self.leader_p_gain * wm.leader.lateral
            return max(-self.max_steer, min(self.max_steer, steer))
        return wm.lane.steering_suggestion

    @staticmethod
    def _make(name: str, speed: float, steering: float, color: LedColor) -> Decision:
        return Decision(
            state_name=name,
            base_speed=speed,
            steering=steering,
            leds={0: color, 2: color, 3: color, 4: color},
        )

    def _make_follow(self, speed: float, steering: float) -> Decision:
        leds = {0: _OFF, 2: _OFF, 3: _OFF, 4: _OFF}
        # Override rear indices with red as the "leader locked" rear signal.
        for idx in self.rear_led_indices:
            leds[int(idx)] = _RED
        return Decision(
            state_name=STATE_FOLLOW,
            base_speed=speed,
            steering=steering,
            leds=leds,
        )


_RED: LedColor = (1.0, 0.0, 0.0)
_YELLOW: LedColor = (1.0, 1.0, 0.0)
_GREEN: LedColor = (0.0, 1.0, 0.0)
_BLUE: LedColor = (0.0, 0.0, 1.0)
_OFF: LedColor = (0.0, 0.0, 0.0)
