from typing import Tuple

from tasks.project.packages.fsm import Decision


def motors_from_decision(d: Decision) -> Tuple[float, float]:
    left = d.base_speed - d.steering
    right = d.base_speed + d.steering
    return _clip01(left), _clip01(right)


def apply_leds(leds, d: Decision) -> None:
    if leds is None:
        return
    for idx, color in d.leds.items():
        try:
            leds.set_rgb(idx, list(color))
        except Exception:
            pass


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
