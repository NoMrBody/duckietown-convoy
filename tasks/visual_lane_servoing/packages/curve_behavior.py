from typing import List, Optional, Tuple
import numpy as np


def detect_curve(yellow_xs: List[int],white_xs:  List[int],curve_threshold: int = 350,
    ) -> Tuple[bool, int]:
    shifts = []

    if len(yellow_xs) >= 2 and yellow_xs[0] is not None and yellow_xs[-1] is not None:
        shifts.append(yellow_xs[-1] - yellow_xs[0])  # far minus near
    if len(white_xs) >= 2 and white_xs[0] is not None and white_xs[-1] is not None:
        shifts.append(white_xs[-1] - white_xs[0])

    if not shifts:
        return False, 0

    avg_shift = int(np.mean(shifts))

    if abs(avg_shift) > curve_threshold:
        # positive shift = lines moved right in image = road curves left
        # negative shift = lines moved left = road curves right
        return True, avg_shift

    return False, 0