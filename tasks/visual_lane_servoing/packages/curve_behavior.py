from typing import List, Tuple
import numpy as np


def detect_curve(
    yellow_xs: List[int],
    white_xs: List[int],
    curve_threshold: int = 350,
) -> Tuple[bool, int]:
    # xs[0] = near (bottom of image), xs[-1] = far (top). Shift = far - near.
    # Positive shift => line bends right ahead; negative => left.
    shifts = []
    if len(yellow_xs) >= 2:
        shifts.append(yellow_xs[-1] - yellow_xs[0])
    if len(white_xs) >= 2:
        shifts.append(white_xs[-1] - white_xs[0])

    if not shifts:
        return False, 0

    shift = float(np.mean(shifts))
    if abs(shift) < curve_threshold:
        return False, 0

    return True, 1 if shift > 0 else -1
