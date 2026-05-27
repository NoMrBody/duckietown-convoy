import os
import time

import yaml

from tasks.project.packages.control import apply_leds, motors_from_decision
from tasks.project.packages.fsm import ConvoyFSM
from tasks.project.packages.perception import Perception

_CONFIG_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml")
)


def _load_cfg() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main(camera, wheels, leds, stop_event):
    cfg = _load_cfg()
    perception = Perception()
    fsm = ConvoyFSM(cfg)

    last_state = None
    last_hw_warn = 0.0
    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.02)
                continue

            now = time.monotonic()
            wm = perception.update(frame, now)
            decision = fsm.step(wm)
            left, right = motors_from_decision(decision)

            try:
                wheels.set_wheels_speed(left, right)
            except OSError as e:
                if now - last_hw_warn > 2.0:
                    print(f"[project] wheels I/O error: {e} (check battery / HAT)")
                    last_hw_warn = now
            try:
                apply_leds(leds, decision)
            except OSError as e:
                if now - last_hw_warn > 2.0:
                    print(f"[project] leds I/O error: {e}")
                    last_hw_warn = now

            if decision.state_name != last_state:
                print(f"[project] state={decision.state_name} "
                      f"speed={decision.base_speed:.2f} steer={decision.steering:+.2f}")
                last_state = decision.state_name
    finally:
        try:
            wheels.set_wheels_speed(0.0, 0.0)
        except Exception:
            pass
        if leds is not None:
            try:
                leds.all_off()
            except Exception:
                pass
