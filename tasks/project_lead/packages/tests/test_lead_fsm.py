"""Pure-logic tests for the lead FSM (no vision / no hardware).

Run:  python -m pytest tasks/project_lead/packages/tests/test_lead_fsm.py
  or:  python tasks/project_lead/packages/tests/test_lead_fsm.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from tasks.project_lead.packages.fsm import (  # noqa: E402
    LeadFSM, STATE_LANE, STATE_STOP, STATE_TURN_R, STATE_TURN_L, STATE_SLOW_AFTER,
    STATE_DONE, STATE_SLOW,
)
from tasks.project.packages.world_model import LaneObs, RedLineObs, SignObs, WorldModel  # noqa: E402


def _lane(healthy=True, steer=0.0):
    return LaneObs(steering_suggestion=steer, base_speed_suggestion=0.0,
                   lane_pixels=600 if healthy else 0, is_curve=False, healthy=healthy)


def _wm(t, lane_healthy=True, signs=None, red=None, steer=0.0):
    return WorldModel(t=t, frame_w=640, frame_h=480, lane=_lane(lane_healthy, steer),
                      leader=None, signs=signs or [], red_line=red)


def _stop():
    return SignObs(kind="STOP", bbox=(0, 0, 10, 10), score=1.0)


def _red(present=True, width=0.6, dist=0.6):
    return RedLineObs(present=present, area_px=1000, width_frac=width, dist_proxy=dist)


def _cfg(**over):
    base = dict(route=["right", "stop"], cruise_speed=0.3, stop_duration=1.0,
                min_turn_s=0.8, max_turn_s=3.0, slow_after_turn_s=2.0,
                stopline_fire_dist=0.45, stopline_fire_width=0.40, stopline_clear_frames=2)
    base.update(over)
    return base


def test_default_is_lane_follow():
    fsm = LeadFSM(_cfg())
    d = fsm.step(_wm(0.0))
    assert d.state_name == STATE_LANE
    assert abs(d.base_speed - 0.3) < 1e-9


def test_slow_sign_slows():
    fsm = LeadFSM(_cfg())
    d = fsm.step(_wm(0.0, signs=[SignObs(kind="SLOW", bbox=(0, 0, 5, 5), score=1.0)]))
    assert d.state_name == STATE_SLOW
    assert d.base_speed < 0.3


def test_stop_sign_remembered_then_stop_then_turn_then_slow_after():
    fsm = LeadFSM(_cfg())

    # sign seen far from the line -> still lane following, but remembered
    assert fsm.step(_wm(0.1, signs=[_stop()])).state_name == STATE_LANE
    assert fsm._sign_pending is True

    # reach the red line (sign gone) -> STOP at sign, route step queued
    d = fsm.step(_wm(0.2, red=_red()))
    assert d.state_name == STATE_STOP and d.base_speed == 0.0
    assert fsm.route_idx == 1

    # still halting; the line is still in view -> no double advance
    d = fsm.step(_wm(0.5, red=_red(), lane_healthy=False))
    assert d.state_name == STATE_STOP
    assert fsm.route_idx == 1  # latch prevents re-fire

    # stop expires -> begin the queued right turn (lane gone mid-intersection)
    d = fsm.step(_wm(1.25, red=_red(present=False), lane_healthy=False))
    assert d.state_name == STATE_TURN_R and d.base_speed > 0.0

    # still turning before min_turn_s
    assert fsm.step(_wm(1.4, lane_healthy=False)).state_name == STATE_TURN_R

    # lane reacquired after min_turn_s -> SLOW_AFTER_TURN + lane-reset request
    d = fsm.step(_wm(2.2, lane_healthy=True))
    assert d.state_name == STATE_SLOW_AFTER
    assert fsm.request_lane_reset is True

    # within the slow-after window
    assert fsm.step(_wm(2.5, lane_healthy=True)).state_name == STATE_SLOW_AFTER

    # window over -> back to lane following
    assert fsm.step(_wm(4.3, lane_healthy=True)).state_name == STATE_LANE


def test_turn_directions_match_lane_convention():
    # Lane convention: negative steering => turn right, positive => turn left.
    right = LeadFSM(_cfg(route=["right", "stop"]))
    d = right.step(_wm(0.2, red=_red(), lane_healthy=False))   # no sign -> immediate maneuver
    assert d.state_name == STATE_TURN_R and d.steering < 0.0

    left = LeadFSM(_cfg(route=["left", "stop"]))
    d = left.step(_wm(0.2, red=_red(), lane_healthy=False))
    assert d.state_name == STATE_TURN_L and d.steering > 0.0


def test_route_exhaustion_final_stop():
    fsm = LeadFSM(_cfg(route=["stop"]))
    # first intersection consumes the only step "stop" (no sign pending here)
    d = fsm.step(_wm(0.2, red=_red()))
    assert d.state_name == STATE_DONE and d.base_speed == 0.0
    # stays done
    assert fsm.step(_wm(0.4)).state_name == STATE_DONE


def test_no_fire_until_line_clears():
    fsm = LeadFSM(_cfg(route=["straight", "straight", "stop"]))
    fsm.step(_wm(0.2, red=_red()))           # fires -> straight maneuver
    idx_after_first = fsm.route_idx
    # red still present next frame: must not advance again
    fsm.step(_wm(0.25, red=_red(), lane_healthy=False))
    assert fsm.route_idx == idx_after_first


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"PASSED {len(fns)} lead-FSM tests")


if __name__ == "__main__":
    _run()
