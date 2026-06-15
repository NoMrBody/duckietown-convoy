from typing import List, Optional, Tuple
import numpy as np


def detect_curve(yellow_xs: List[Optional[int]], white_xs: List[Optional[int]],
                 curve_threshold: int = 15,
                 bend_max: int = 90,
                 monotonic_tol: int = 8,
    ) -> Tuple[bool, int]:
    """Per-line bend across the three slices. A straight road line projects to
    a straight image line, and the slice rows are equally spaced, so
    x0 - 2*x1 + x2 ~= 0 on a straight for ANY bot pose (the near-far shift,
    by contrast, has a large perspective-convergence baseline even when the
    road is dead straight). A curve bends the projected line.

    Needs the line visible in all three slices; lists are slice-aligned with
    None placeholders. Sign is flipped so positive keeps meaning "road curves
    left" (the old near-far convention).

    Robustness gate (sim showed |bend| ~ 300 on straights): a per-slice pick
    set only describes a real road line if its three picks lie on ONE
    continuous arc. That arc is MONOTONIC in x across the equally-spaced rows
    (it bends one way) and its second difference is modest (genuine curves
    measure ~15-50 px). Picks that ZIGZAG (opposite-signed inter-row steps) or
    yield an implausibly large bend are a detection artifact -- e.g. the yellow
    centerline pick averaging in an off-road blob, so it jumps between the
    blob and the real dash slice to slice. Such picks are REJECTED rather than
    slammed into CURVE; the bot would otherwise be pinned in CURVE forever.
    """
    bends = []
    for xs in (yellow_xs, white_xs):
        if len(xs) != 3 or any(x is None for x in xs):
            continue
        x0, x1, x2 = xs
        bend = x0 - 2 * x1 + x2
        # Implausibly large second difference -> the picks are not one line.
        if abs(bend) > bend_max:
            continue
        # Non-monotonic zigzag beyond a small noise tolerance -> not a single
        # continuous arc (a straight or a real curve never reverses direction
        # across three adjacent rows).
        d01, d12 = x1 - x0, x2 - x1
        if d01 * d12 < 0 and min(abs(d01), abs(d12)) > monotonic_tol:
            continue
        bends.append(bend)

    if not bends:
        return False, 0

    avg_bend = -float(np.mean(bends))  # left curve: far slice pulled left -> raw bend < 0

    if abs(avg_bend) > curve_threshold:
        return True, int(avg_bend)

    return False, 0
