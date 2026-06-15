"""End-to-end curve-detector regression tests.

Exercises the REAL pipeline on synthetic 640x480 BGR frames that mimic the
Godot pinhole sim camera (no lens distortion):

    detect_lane_markings  ->  detect_lines_in_slices  ->  detect_curve

Regression target: the LEAD bot was pinned in the FSM "CURVE" state on
straight roads. Root cause (detector half): the white side-filter
_keep_blobs_on_side(white_mask,'left',0.55) amputated the bot's OWN right lane
edge — the exact blob the per-slice picker hunts for — so on degraded sim
frames the picker latched a wandering left-half blob and fabricated a large
x0-2x1+x2 bend. Fix 1 removed that filter so both real white edges survive and
the picker tracks ONE edge consistently (bend~0 on a straight).

The make_frame/run_one helpers are promoted from /tmp/repro_curve.py so the
reproduction is committed and runnable.

Run:  python -m pytest tasks/visual_lane_servoing/packages/tests/test_curve_regression.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from tasks.visual_lane_servoing.packages.visual_servoing_activity import detect_lane_markings  # noqa: E402
from tasks.visual_lane_servoing.packages.agent import detect_lines_in_slices  # noqa: E402
from tasks.visual_lane_servoing.packages.curve_behavior import detect_curve  # noqa: E402

W, H = 640, 480
VP_Y = 150            # vanishing point row (above the crop_top=192 line)
HORIZON_BOTTOM = 480  # road lines drawn from VP down to bottom
CURVE_THRESHOLD = 15  # mirrors config/lane_servoing_config.yaml

# Colors (BGR)
ASPHALT = (90, 90, 90)
YELLOW = (0, 210, 230)   # HSV hue ~30
WHITE = (235, 235, 235)


def _lerp_x(x_vp, x_bottom, y):
    """x of a straight road line at row y, going VP(VP_Y) -> bottom(H)."""
    t = (y - VP_Y) / float(H - VP_Y)
    return x_vp + (x_bottom - x_vp) * t


def make_frame(offset_px=0, jitter=0, curve_px=0, seed=0,
               right_bottom=600, left_bottom=-60, white_w=6):
    """Build a 640x480 BGR straight (or curved) road, right-lane bot view.

    offset_px: lateral shift of whole road (+ = road shifts right => bot left).
    curve_px : if !=0, bend the lines toward one side near the top (CONTROL).
    right_bottom/left_bottom: bottom-row x of the two white edges (controls how
        wide the lane is; a narrower lane lets the right edge sit nearer center).
    """
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), ASPHALT, dtype=np.uint8)
    # Mild noise so Sobel gradients exist at marking edges.
    noise = rng.normal(0, 6, (H, W, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Right-lane bot view: the yellow centerline is LEFT of frame-center and
    # the bot's own (right) white edge is to the RIGHT. The far/left lane's
    # outer white edge is mostly off-frame. All three converge to a vanishing
    # point near top-center.
    vp = W // 2 + offset_px // 3  # vp shifts less than bottom under lateral move
    base = {
        'left_white':  (vp - 70, left_bottom + offset_px),   # opposite-lane outer edge
        'yellow':      (vp,       250 + offset_px),           # centerline (left of bot)
        'right_white': (vp + 70,  right_bottom + offset_px),  # bot's own right lane edge
    }

    def draw_line(x_vp, x_bottom, color, dashed=False, width=6):
        ys = np.arange(VP_Y, HORIZON_BOTTOM, 1)
        for y in ys:
            j = rng.normal(0, jitter) if jitter else 0.0
            bend = 0.0
            if curve_px:
                tt = (HORIZON_BOTTOM - y) / float(HORIZON_BOTTOM - VP_Y)
                bend = curve_px * tt * tt
            x = int(round(_lerp_x(x_vp, x_bottom, y) + j + bend))
            if dashed:
                if ((y - VP_Y) // 24) % 2 == 1:
                    continue
            cv2.circle(img, (x, y), width // 2, color, -1)

    draw_line(*base['left_white'], WHITE, dashed=False, width=white_w)
    draw_line(*base['yellow'], YELLOW, dashed=True, width=7)
    draw_line(*base['right_white'], WHITE, dashed=False, width=white_w)
    return img


def run_one(img):
    """Run the REAL pipeline; return (is_curve, yellow_xs, white_xs)."""
    mask_left, mask_right = detect_lane_markings(img)
    mask_y = (mask_left * 255).astype(np.uint8)
    mask_w = (mask_right * 255).astype(np.uint8)
    h = mask_y.shape[0]
    yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
    is_curve, _cdir = detect_curve(yellow_xs, white_xs, CURVE_THRESHOLD)
    return is_curve, yellow_xs, white_xs


def _bend(xs):
    if len(xs) == 3 and all(x is not None for x in xs):
        return xs[0] - 2 * xs[1] + xs[2]
    return None


# --- tests ------------------------------------------------------------------

def test_straight_not_curve():
    """A straight road must NOT read as a curve at any reasonable lateral
    offset or with mild jitter."""
    scen = 0
    for offset in (-60, -40, -20, 0, 20, 40, 60, 90, 120):
        for jitter in (0, 1.5):
            img = make_frame(offset_px=offset, jitter=jitter, curve_px=0, seed=100 + scen)
            is_curve, yxs, wxs = run_one(img)
            assert is_curve is False, (
                f"FALSE curve on straight off={offset} jit={jitter}: "
                f"yellow_xs={yxs} white_xs={wxs} bend_y={_bend(yxs)} bend_w={_bend(wxs)}")
            scen += 1


def test_straight_narrow_lane_not_curve():
    """Narrow-lane straights (the bot's own right edge nearer center, so both
    white edges are well in-frame) must stay clean once Fix 1 lets both edges
    survive the mask."""
    for rb in (560, 520, 480, 440, 420, 400):
        for off in (0, 30):
            img = make_frame(offset_px=off, jitter=0.8, curve_px=0, seed=900 + rb + off,
                             right_bottom=rb, left_bottom=20)
            is_curve, yxs, wxs = run_one(img)
            assert is_curve is False, (
                f"FALSE curve on narrow straight rb={rb} off={off}: "
                f"yellow_xs={yxs} white_xs={wxs} bend_y={_bend(yxs)} bend_w={_bend(wxs)}")


def test_curve_is_detected():
    """A genuinely bent road must be detected. |curve_px|>=360 is used because
    cpx=200 yields bend~11 < threshold and is legitimately below detection."""
    for curve_px in (360, -360, 480, -480):
        img = make_frame(offset_px=0, jitter=0, curve_px=curve_px, seed=500 + curve_px)
        is_curve, yxs, wxs = run_one(img)
        assert is_curve is True, (
            f"MISSED genuine curve curve_px={curve_px}: "
            f"yellow_xs={yxs} white_xs={wxs} bend_y={_bend(yxs)} bend_w={_bend(wxs)}")


def test_white_edge_consistent_on_straight():
    """Guards Fix 1 specifically: on straights where BOTH white edges are
    in-frame, the picker must track ONE edge consistently across all three
    slices — never jump between the two edges. So white_xs is either all-None
    or its second difference |x0-2*x1+x2| stays within the curve threshold.

    This FAILS on the pre-fix mask, which amputated the bot's own right edge
    and let a spurious left-half blob be picked, producing a large jumpy bend.
    """
    for offset in (40, 60, 90, 120):
        for rb in (600, 520, 440):
            img = make_frame(offset_px=offset, jitter=0.0, curve_px=0,
                             seed=300 + offset + rb, right_bottom=rb, left_bottom=20)
            _is_curve, _yxs, wxs = run_one(img)
            if all(x is not None for x in wxs):
                bw = _bend(wxs)
                assert abs(bw) <= CURVE_THRESHOLD, (
                    f"white edge JUMPED between slices on straight "
                    f"off={offset} rb={rb}: white_xs={wxs} bend_w={bw}")


def test_offroad_yellow_blob_not_curve():
    """Real-sim failure mode (observed: CURVE=True dir=-304 on a straight).

    The broad yellow hue (H<=65) catches the off-road grass/dirt as a big
    yellow blob on the road shoulder. The old yellow pick = mean(all clusters)
    averaged that blob into the centerline estimate, so the per-slice picks
    zigzagged hundreds of px and detect_curve fired. The wide-cluster reject in
    detect_lines_in_slices plus the bend sanity/monotonic gate must keep a
    straight from reading as a curve even with the blob present.
    """
    OLIVE = (40, 150, 170)  # BGR; HSV hue ~ 27-33 with high S -> inside yellow band
    scen = 0
    for offset in (-40, 0, 40, 90):
        img = make_frame(offset_px=offset, jitter=1.0, curve_px=0, seed=700 + scen)
        # Paint a large off-road wedge on the right shoulder (grass caught as yellow).
        pts = np.array([[W, H], [W, 230], [int(W * 0.62), 230], [int(W * 0.80), H]], np.int32)
        cv2.fillPoly(img, [pts], OLIVE)
        is_curve, yxs, wxs = run_one(img)
        assert is_curve is False, (
            f"FALSE curve from off-road yellow blob off={offset}: "
            f"yellow_xs={yxs} white_xs={wxs} bend_y={_bend(yxs)} bend_w={_bend(wxs)}")
        scen += 1


def test_detect_curve_rejects_garbage_picks():
    """Unit-level gate: implausibly large or zigzagging picks are NOT a curve."""
    # The exact real-sim symptom: a huge bend from picks that don't form a line.
    assert detect_curve([275, 400, 150], [None, None, None], 15)[0] is False
    # Huge monotonic bend (still implausible for one road line) -> rejected.
    assert detect_curve([400, 80, 360], [None, None, None], 15)[0] is False
    # Non-monotonic zigzag within threshold-ish but reversing -> rejected.
    assert detect_curve([200, 260, 205], [None, None, None], 15)[0] is False


def test_curve_steers_correct_direction():
    """Regression for the off-road bug: a RIGHT curve must steer RIGHT and a
    LEFT curve must steer LEFT. The picker once flipped to the opposite-lane
    white edge mid-frame on a curve, so the bot steered the wrong way and ran
    off the road. Convention: steering_suggestion>0 turns LEFT, <0 turns RIGHT.
    """
    from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

    def steer_for(curve_px):
        img = make_frame(offset_px=0, jitter=0, curve_px=curve_px, seed=42)
        ag = LaneServoingAgent()
        left = right = 0.0
        for _ in range(6):  # let the EMA/PD settle
            left, right = ag.compute_commands(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return (right - left) / 2.0

    assert steer_for(360) < 0, "RIGHT curve must steer right (was driving off-road left)"
    assert steer_for(-360) > 0, "LEFT curve must steer left"


def test_white_edge_single_on_curve():
    """The white edge picks must stay on ONE continuous edge across the three
    slices on a curve (no jump between the road's two white edges)."""
    for curve_px in (360, -360):
        _is_curve, _yxs, wxs = run_one(make_frame(curve_px=curve_px, seed=7))
        if all(x is not None for x in wxs):
            d0, d1 = wxs[1] - wxs[0], wxs[2] - wxs[1]
            assert d0 * d1 >= 0 or min(abs(d0), abs(d1)) <= 8, (
                f"white edge jumped between road edges on curve_px={curve_px}: {wxs}")


def test_detect_curve_accepts_real_curve():
    """A monotonic, modest-magnitude bend is a genuine curve."""
    # Left curve: far slice pulled left, monotonic, bend within sanity bound.
    ok, d = detect_curve([300, 330, 380], [None, None, None], 15)
    assert ok is True and d != 0
    # Right curve, monotonic the other way.
    ok2, _ = detect_curve([380, 330, 300], [None, None, None], 15)
    assert ok2 is True


if __name__ == "__main__":
    test_straight_not_curve()
    test_straight_narrow_lane_not_curve()
    test_curve_is_detected()
    test_white_edge_consistent_on_straight()
    test_offroad_yellow_blob_not_curve()
    test_detect_curve_rejects_garbage_picks()
    test_curve_steers_correct_direction()
    test_white_edge_single_on_curve()
    test_detect_curve_accepts_real_curve()
    print("PASSED curve regression tests")
