from typing import Tuple

from tasks.project.packages.fsm_common import Decision


def motors_from_decision(d: Decision) -> Tuple[float, float]:
    # Clamp steering to the base speed so a turn arcs forward instead of
    # collapsing into a single-wheel pivot; base==0 still yields a full stop.
    base = d.base_speed
    steer = max(-base, min(base, d.steering))
    left = base - steer
    right = base + steer
    return _clip01(left), _clip01(right)


def apply_leds(leds, d: Decision) -> None:
    if leds is None:
        return
    for idx, color in d.leds.items():
        try:
            leds.set_rgb(idx, list(color))
        except Exception:
            pass


def blink_all(leds, color, now, hz=2.0) -> None:
    """Blink all LEDs at `hz`. `color` is RGB in [0,1]; off on the dark phase."""
    if leds is None:
        return
    on = int(now * hz * 2) % 2 == 0
    rgb = list(color) if on else [0.0, 0.0, 0.0]
    for idx in (0, 2, 3, 4):
        try:
            leds.set_rgb(idx, rgb)
        except Exception:
            pass


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
