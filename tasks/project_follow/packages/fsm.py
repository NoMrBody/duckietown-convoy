"""Follower-bot finite state machine: pure pursuit of the lead's circle-grid
back marker, with a lane-following fallback. No traffic-sign logic at all -- the
lead owns that. The follower stops *implicitly* when the lead halts (the grid
grows -> distance taper -> speed 0).

States:
  WAIT_LEAD  - startup: hold until the lead has opened a gap (don't lurch into a
               stationary lead at boot).
  FOLLOW     - marker present: distance-tapered speed, steer toward the marker.
  CLOSE_STOP - marker lost while close: STOP (don't coast into the lead).
  REACQUIRE  - marker lost within grace: slow creep + steer toward the last-seen
               bearing, to turn *into* the corner the lead took.
  LANE_FOLLOW- marker lost past grace, lane healthy: cruise on the lane.
  HOLD       - nothing visible: stop.

Pure logic over a WorldModel -> unit-testable without vision/hardware.
"""
from typing import Optional

from tasks.project.packages.fsm_common import (
    BLUE, OFF, RED, YELLOW, Decision, all_leds, clamp, follow_speed, lateral_to_steer,
)
from tasks.project.packages.world_model import WorldModel

STATE_WAIT       = "WAIT_LEAD"
STATE_FOLLOW     = "FOLLOW"
STATE_CLOSE_STOP = "CLOSE_STOP"
STATE_REACQUIRE  = "REACQUIRE"
STATE_LANE       = "LANE_FOLLOW"
STATE_HOLD       = "HOLD"


class FollowerFSM:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.cruise_speed = float(cfg.get("cruise_speed", 0.3))
        # Distance proxy = circle-grid mean dot spacing (px). Grows as lead nears.
        self.grid_safe_px = float(cfg.get("grid_safe_px", 18))   # <= this -> full speed (far)
        self.grid_stop_px = float(cfg.get("grid_stop_px", 70))   # >= this -> zero speed (close)
        self.grid_close_px = float(cfg.get("grid_close_px", 55))  # lost-while-this-close -> STOP, don't coast
        self.grid_arm_px = float(cfg.get("grid_arm_px", self.grid_stop_px))  # arm once a gap this big exists
        self.grid_min_score = float(cfg.get("grid_min_score", 0.5))

        self.leader_p_gain = float(cfg.get("leader_p_gain", 0.6))
        self.max_steer = float(cfg.get("max_steer", 0.4))
        self.leader_lost_grace_s = float(cfg.get("leader_lost_grace_s", 1.0))

        self.reacquire_creep = float(cfg.get("reacquire_creep_speed", 0.12))
        self.reacquire_steer_gain = float(cfg.get("reacquire_steer_gain", 0.8))
        self.heading_gain = float(cfg.get("reacquire_heading_gain", 0.0))  # best-effort; off by default

        self.rear_led_indices = list(cfg.get("rear_led_indices", [3, 4]))

        self._last_leader_t = -1e9
        self._last_lateral = 0.0
        self._last_span: Optional[float] = None
        self._last_heading: Optional[float] = None
        self._armed = False

    def step(self, wm: WorldModel) -> Decision:
        t = wm.t
        leader = wm.leader
        confident = leader is not None and leader.score >= self.grid_min_score

        if confident:
            self._last_leader_t = t
            self._last_lateral = leader.lateral
            self._last_span = leader.pair_px if leader.pair_px is not None else leader.distance_px
            self._last_heading = leader.heading
            if self._last_span is not None and self._last_span <= self.grid_arm_px:
                self._armed = True  # a real gap has opened -> safe to follow

        # Startup: don't drive until the lead has opened a gap at least once.
        if not self._armed:
            return self._mk(STATE_WAIT, 0.0, 0.0, all_leds(RED if confident else OFF))

        if confident:
            span = self._last_span if self._last_span is not None else self.grid_safe_px
            speed = follow_speed(span, self.grid_safe_px, self.grid_stop_px, self.cruise_speed)
            steer = lateral_to_steer(leader.lateral, self.leader_p_gain, self.max_steer)
            return self._mk(STATE_FOLLOW, speed, steer, self._rear_signal())

        # Marker lost.
        within_grace = (t - self._last_leader_t) < self.leader_lost_grace_s
        if within_grace:
            # Stop-floor: lost while close (grid drops out at close range) -> STOP, not coast.
            if self._last_span is not None and self._last_span >= self.grid_close_px:
                return self._mk(STATE_CLOSE_STOP, 0.0, 0.0, all_leds(RED))
            # Creep toward where the lead was, to turn into the corner it took.
            steer = lateral_to_steer(self._last_lateral, self.reacquire_steer_gain, self.max_steer)
            if self.heading_gain and self._last_heading is not None:
                steer = clamp(steer + self.heading_gain * self._last_heading,
                              -self.max_steer, self.max_steer)
            return self._mk(STATE_REACQUIRE, self.reacquire_creep, steer, all_leds(YELLOW))

        # Grace expired.
        if wm.lane.healthy:
            return self._mk(STATE_LANE, self.cruise_speed, wm.lane.steering_suggestion, all_leds(BLUE))
        return self._mk(STATE_HOLD, 0.0, 0.0, all_leds(OFF))

    def _rear_signal(self):
        leds = {0: OFF, 1: OFF, 2: OFF, 3: OFF, 4: OFF}
        for idx in self.rear_led_indices:
            leds[int(idx)] = RED  # signal to any further follower in the chain
        return leds

    @staticmethod
    def _mk(name: str, speed: float, steering: float, leds: dict) -> Decision:
        return Decision(state_name=name, base_speed=speed, steering=steering, leds=leds)
