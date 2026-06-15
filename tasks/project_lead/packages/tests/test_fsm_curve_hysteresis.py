"""Pure-logic regression test for the curve-latch ENTRY hysteresis (Fix 2).

No vision / no hardware: drives LeadFSM.step() with a WorldModel whose
wm.lane.is_curve is scripted, and checks the resulting state.

Regression target: the FSM block-7 latch re-armed the full curve_hold_s (0.4s)
window on a SINGLE is_curve=True frame. At ~30fps that hold spans ~12 frames,
so even an ~8% sporadic false-positive rate (1 True every 12 frames) kept
CURVE asserted forever and LANE_FOLLOW was never reached. Fix 2 added a
sliding-window entry vote (>=curve_enter_frac of a curve_enter_window_s window
must be True to ENTER), which defeats the single-frame latch while still
entering on a genuine sustained curve.

Run:  python -m pytest tasks/project_lead/packages/tests/test_fsm_curve_hysteresis.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from tasks.project_lead.packages.fsm import LeadFSM, STATE_LANE, STATE_CURVE  # noqa: E402
from tasks.project.packages.world_model import LaneObs, WorldModel  # noqa: E402

DT = 0.033  # ~30 fps, matches the virtual server stream cadence


def _wm(t, is_curve):
    lane = LaneObs(steering_suggestion=0.0, base_speed_suggestion=0.0,
                   lane_pixels=600, is_curve=is_curve, healthy=True)
    return WorldModel(t=t, frame_w=640, frame_h=480, lane=lane,
                      leader=None, signs=[], red_line=None)


def _cfg(**over):
    # Healthy lane, no signs / red lines, so block 7 (curve) and block 8
    # (lane follow) are the only reachable states.
    base = dict(route=["stop"], cruise_speed=0.3,
                curve_hold_s=0.4, curve_enter_window_s=0.30,
                curve_enter_frac=0.6, curve_enter_min_samples=4)
    base.update(over)
    return base


def test_sporadic_false_curve_does_not_latch():
    """1 spurious True every 12 frames (~8%) over ~3s must NOT pin CURVE; the
    FSM must reach LANE_FOLLOW. Proves entry hysteresis defeats the latch."""
    fsm = LeadFSM(_cfg())
    n_steps = int(3.0 / DT)  # ~90 frames
    states = []
    for i in range(n_steps):
        t = i * DT
        is_curve = (i % 12 == 0)  # ~8% duty
        states.append(fsm.step(_wm(t, is_curve)).state_name)

    # It must spend the clear majority of frames in LANE_FOLLOW and clearly
    # not be permanently stuck in CURVE.
    assert STATE_LANE in states, f"never reached LANE_FOLLOW: {set(states)}"
    n_curve = states.count(STATE_CURVE)
    assert n_curve < n_steps * 0.25, (
        f"CURVE asserted on {n_curve}/{n_steps} frames despite only ~8% "
        f"spurious detections — single-frame latch not defeated")
    # And the steady-state tail (after the window has flushed) is LANE_FOLLOW.
    assert states[-1] == STATE_LANE, f"ended stuck in {states[-1]}"


def test_sustained_curve_enters_curve_state():
    """A genuine sustained curve (is_curve True every frame) must ENTER CURVE
    within ~curve_enter_window_s (~0.3s ~ 9 frames)."""
    fsm = LeadFSM(_cfg())
    entered_at = None
    for i in range(40):
        t = i * DT
        d = fsm.step(_wm(t, True))
        if d.state_name == STATE_CURVE and entered_at is None:
            entered_at = t
            break
    assert entered_at is not None, "sustained curve never entered STATE_CURVE"
    assert entered_at <= 0.30 + 1e-6, f"entered CURVE too late at t={entered_at:.3f}s"


def test_majority_window_enters_curve():
    """>=60% True over a sustained window enters CURVE even with some flicker
    (proves the vote, not a single frame, governs entry)."""
    fsm = LeadFSM(_cfg())
    # 2 True : 1 False repeating == ~67% duty, above curve_enter_frac=0.6.
    saw_curve = False
    for i in range(60):
        t = i * DT
        is_curve = (i % 3 != 0)  # True, True, False, ...
        if fsm.step(_wm(t, is_curve)).state_name == STATE_CURVE:
            saw_curve = True
            break
    assert saw_curve, "sustained ~67%-duty curve never entered STATE_CURVE"


if __name__ == "__main__":
    test_sporadic_false_curve_does_not_latch()
    test_sustained_curve_enters_curve_state()
    test_majority_window_enters_curve()
    print("PASSED fsm curve hysteresis tests")
