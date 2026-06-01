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


def main(camera, wheels, leds, stop_event, frame_queue=None, debug_lock=None, debug_dict=None):
    cfg = _load_cfg()
    perception = Perception()
    fsm = ConvoyFSM(cfg)

    last_state = None
    last_hw_warn = 0.0
    last_dbg = 0.0
    last_fps_update = 0.0
    frame_count = 0
    fps = 0.0

    try:
        while not stop_event.is_set():
            loop_start = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Share frame with visualizer
            if frame_queue is not None:
                try:
                    # Non-blocking put - drop if queue is full
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass

            now = time.monotonic()
            wm = perception.update(frame, now)
            decision = fsm.step(wm)
            left, right = motors_from_decision(decision)

            # Calculate FPS
            frame_count += 1
            if now - last_fps_update > 1.0:
                fps = frame_count / (now - last_fps_update)
                frame_count = 0
                last_fps_update = now

            # Update shared debug info for visualization
            if debug_lock is not None and debug_dict is not None:
                dbg = perception.last_debug_info
                with debug_lock:
                    debug_dict.update({
                        'state': decision.state_name,
                        'base_speed': decision.base_speed,
                        'steering': decision.steering,
                        'left_speed': left,
                        'right_speed': right,
                        'leader_source': dbg.get('leader_source'),
                        'led_pair_px': dbg.get('led_pair_px'),
                        'apriltag_ids': dbg.get('apriltag_ids', []),
                        'fps': fps,
                    })

            if now - last_dbg > 0.5:
                dbg = perception.last_debug_info
                print(f"[project] {decision.state_name} "
                      f"pair_px={dbg.get('led_pair_px')} src={dbg.get('leader_source')} "
                      f"tags={dbg.get('apriltag_ids')} "
                      f"base={decision.base_speed:.2f} steer={decision.steering:+.2f} "
                      f"L={left:.2f} R={right:.2f}")
                last_dbg = now

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
