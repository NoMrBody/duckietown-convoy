"""Pure-logic tests for the lead FSM (no vision / no hardware).

Run:  python -m pytest tasks/project_lead/packages/tests/test_lead_fsm.py
  or:  python tasks/project_lead/packages/tests/test_lead_fsm.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from tasks.project_lead.packages.fsm import (  # noqa: E402
    LeadFSM, STATE_LANE, STATE_STOP, STATE_TURN_R, STATE_TURN_L, STATE_SLOW_AFTER,
    STATE_DONE, STATE_SLOW, STATE_CROSS,
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


def test_one_line_does_not_burn_route_during_maneuver():
    # Regression for the route-burn latch bug: a single physical red line that
    # flickers out of view DURING the turn / slow-after window must not re-arm
    # the latch and consume extra route steps (which would reach terminal 'stop').
    fsm = LeadFSM(_cfg(route=["right", "straight", "stop"], stopline_clear_frames=2,
                       min_turn_s=0.8, slow_after_turn_s=2.0))

    # First (and only physical) intersection: fire -> right turn.
    fsm.step(_wm(0.2, red=_red(), lane_healthy=False))
    assert fsm.route_idx == 1

    # Turn completes once the lane is reacquired -> SLOW_AFTER window opens.
    assert fsm.step(_wm(1.1, lane_healthy=True)).state_name == STATE_SLOW_AFTER

    # The SAME line drops out of the band for >= clear_frames during slow-after.
    fsm.step(_wm(1.5, red=_red(present=False), lane_healthy=True))
    fsm.step(_wm(1.9, red=_red(present=False), lane_healthy=True))

    # Slow-after ends with the same line back in view: must NOT advance again.
    fsm.step(_wm(3.5, red=_red(), lane_healthy=True))
    assert fsm.route_idx == 1, "single red line re-fired and burned the route"


def test_cross_exits_on_distance():
    # Closed-loop straight cross exits when the encoder distance reaches target,
    # well before the max_cross_s safety timeout.
    fsm = LeadFSM(_cfg(route=["straight", "stop"], cross_distance_m=0.35, max_cross_s=99.0))
    d = fsm.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    assert d.state_name == STATE_CROSS
    d = fsm.step(_wm(0.4, lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.20)
    assert d.state_name == STATE_CROSS                       # not yet at target
    d = fsm.step(_wm(0.6, lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.35)
    assert d.state_name == STATE_SLOW_AFTER                  # distance reached -> exit


def test_cross_holds_heading_sign():
    # Heading-hold during a cross must steer AGAINST the drift (control.py sign).
    fsm = LeadFSM(_cfg(route=["straight", "stop"], cross_distance_m=1.0, max_cross_s=99.0))
    fsm.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    d = fsm.step(_wm(0.4, lane_healthy=False), turn_yaw_rad=0.5, fwd_dist_m=0.1)
    assert d.state_name == STATE_CROSS and d.steering < 0.0  # drifting left -> steer right

    fsm2 = LeadFSM(_cfg(route=["straight", "stop"], cross_distance_m=1.0, max_cross_s=99.0))
    fsm2.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    d = fsm2.step(_wm(0.4, lane_healthy=False), turn_yaw_rad=-0.5, fwd_dist_m=0.1)
    assert d.state_name == STATE_CROSS and d.steering > 0.0  # drifting right -> steer left


def test_turn_exits_on_yaw_target():
    # Right turn: yaw goes negative; exits when |yaw| reaches the heading target.
    fsm = LeadFSM(_cfg(route=["right", "stop"], turn_yaw_target_rad=1.40, max_turn_s=99.0))
    d = fsm.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    assert d.state_name == STATE_TURN_R and d.steering < 0.0
    d = fsm.step(_wm(0.5, lane_healthy=False), turn_yaw_rad=-0.7, fwd_dist_m=0.0)
    assert d.state_name == STATE_TURN_R                      # not yet at target
    d = fsm.step(_wm(0.8, lane_healthy=False), turn_yaw_rad=-1.4, fwd_dist_m=0.0)
    assert d.state_name == STATE_SLOW_AFTER                  # heading reached -> exit

    left = LeadFSM(_cfg(route=["left", "stop"], turn_yaw_target_rad=1.40, max_turn_s=99.0))
    d = left.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    assert d.state_name == STATE_TURN_L and d.steering > 0.0


def test_turn_p_control_tapers():
    # The turn steer tapers as the remaining yaw error shrinks, with a floor.
    fsm = LeadFSM(_cfg(route=["left", "stop"], turn_yaw_target_rad=1.40, max_turn_s=99.0,
                       turn_steer=0.25, turn_kp=0.8))
    fsm.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    s_early = fsm.step(_wm(0.4, lane_healthy=False), turn_yaw_rad=0.3, fwd_dist_m=0.0).steering
    s_late  = fsm.step(_wm(0.6, lane_healthy=False), turn_yaw_rad=1.2, fwd_dist_m=0.0).steering
    assert abs(s_late) < abs(s_early)                        # tapers toward target
    assert abs(s_late) >= 0.10 - 1e-9                        # never below the floor


def test_no_odometry_falls_back_to_timed():
    # With no encoder scalars, behaviour is the legacy timed / lane-reacquire path.
    fsm = LeadFSM(_cfg(route=["straight", "stop"], min_cross_s=0.4, max_cross_s=3.0))
    assert fsm.step(_wm(0.2, red=_red(), lane_healthy=False)).state_name == STATE_CROSS
    assert fsm.step(_wm(0.5, lane_healthy=True)).state_name == STATE_CROSS      # before min_cross_s
    assert fsm.step(_wm(0.7, lane_healthy=True)).state_name == STATE_SLOW_AFTER  # lane reacquired


def test_max_timeout_overrides_odometry():
    # Even with no forward progress, the cross exits at the max_cross_s safety net.
    fsm = LeadFSM(_cfg(route=["straight", "stop"], cross_distance_m=99.0, max_cross_s=1.0))
    fsm.step(_wm(0.2, red=_red(), lane_healthy=False), turn_yaw_rad=0.0, fwd_dist_m=0.0)
    assert fsm.step(_wm(0.5, lane_healthy=False),
                    turn_yaw_rad=0.0, fwd_dist_m=0.0).state_name == STATE_CROSS
    assert fsm.step(_wm(1.3, lane_healthy=False),
                    turn_yaw_rad=0.0, fwd_dist_m=0.0).state_name == STATE_SLOW_AFTER


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"PASSED {len(fns)} lead-FSM tests")


if __name__ == "__main__":
    _run()
