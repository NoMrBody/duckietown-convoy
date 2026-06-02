import sys
import os
import signal
import threading
import argparse
import queue

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, request
import numpy as np
import cv2

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import shutdown_cleanup, suppress_http_logs

import tasks.project_follow.packages.agent as agent

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Convoy — Follower</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
  header { padding:12px 16px; background:#1a1a1a; border-bottom:1px solid #333; }
  main { display:flex; justify-content:center; padding:16px; }
  img.stream { max-width:100%; height:auto; border:1px solid #333; background:#000; }
</style></head>
<body>
  <header>Convoy — Follower Bot</header>
  <main><img class="stream" src="/video" alt="camera"></main>
</body></html>
"""

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()

_frame_queue = queue.Queue(maxsize=2)
_debug_info_lock = threading.Lock()
_debug_info = {}

# Display stream: downscale + lower JPEG quality so frames are ~4x smaller over
# wifi (cuts latency), and always send the freshest frame (drain the queue).
_STREAM_SIZE = (480, 360)   # (width, height) of the streamed image
_STREAM_QUALITY = 45        # JPEG quality [0..100] for the stream

# Optional 2x2 debug montage: annotated camera + yellow / white lane masks and
# the circle-grid leader detection, so HSV / grid tuning can be done from the
# browser. Each panel is _PANEL_SIZE, giving a 640x480 montage. Falls back to
# the plain feed if the vision modules can't be imported. NOTE: lane masks
# reflect the HSV config loaded at startup — redeploy to pick up edits.
_SHOW_MASKS = True
_PANEL_SIZE = (320, 240)    # (w, h) per panel
try:
    import yaml
    from tasks.visual_lane_servoing.packages.visual_servoing_activity import detect_lane_markings
    from tasks.project.packages.marker_grid import MarkerGridTracker
    _follow_cfg_file = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "project_follow_config.yaml"))
    try:
        with open(_follow_cfg_file) as _cf:
            _follow_cfg = yaml.safe_load(_cf) or {}
    except Exception:
        _follow_cfg = {}
    _grid_tracker = MarkerGridTracker(cfg=_follow_cfg)
    _MASKS_AVAILABLE = True
except Exception as _mask_err:
    print(f"[follow] mask preview disabled: {_mask_err}")
    _MASKS_AVAILABLE = False


def _mask_to_bgr(mask, color, label):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask > 0] = color
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _grid_panel(panel):
    vis = panel.copy()
    try:
        obs = _grid_tracker.update(panel)
        if obs is not None:
            x1, y1, x2, y2 = obs.bbox
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cx, cy = obs.midpoint
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    except Exception:
        pass
    cv2.putText(vis, "GRID (leader)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def _montage(frame_bgr):
    panel = cv2.resize(frame_bgr, _PANEL_SIZE, interpolation=cv2.INTER_AREA)
    cam = visualize(panel)
    z = np.zeros((_PANEL_SIZE[1], _PANEL_SIZE[0]), dtype=np.uint8)
    try:
        m_yellow, m_white = detect_lane_markings(panel)
    except Exception:
        m_yellow = m_white = z
    yellow = _mask_to_bgr(m_yellow, (0, 255, 255), "YELLOW centerline")
    white  = _mask_to_bgr(m_white,  (255, 255, 255), "WHITE edge")
    grid   = _grid_panel(panel)
    top    = cv2.hconcat([cam, yellow])
    bottom = cv2.hconcat([white, grid])
    return cv2.vconcat([top, bottom])


def visualize(frame_bgr):
    if frame_bgr is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank

    display = frame_bgr.copy()
    h, w = display.shape[:2]
    with _debug_info_lock:
        info = _debug_info.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    state = info.get('state', 'INIT')
    cv2.putText(display, f"State: {state}", (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Base: {info.get('base_speed', 0.0):.2f} "
                         f"Steer: {info.get('steering', 0.0):+.2f}", (10, 50), font, 0.5, (0, 255, 0), 1)
    cv2.putText(display, f"L: {info.get('left_speed', 0.0):+.2f}  "
                         f"R: {info.get('right_speed', 0.0):+.2f}", (10, 70), font, 0.5, (0, 255, 0), 1)

    src = info.get('leader_source')
    if src:
        span = info.get('led_pair_px')
        text = f"Leader: {src}"
        if span:
            text += f" (span={span}px)"
        hdg = info.get('grid_heading')
        if hdg is not None:
            text += f" hdg={hdg}"
        cv2.putText(display, text, (10, h - 40), font, 0.5, (255, 255, 0), 1)
    cv2.putText(display, f"{info.get('fps', 0.0):.1f} FPS", (w - 100, 25), font, 0.5, (200, 200, 200), 1)
    return display


def generate_frames():
    import time
    while True:
        try:
            frame = _frame_queue.get(timeout=0.5)
            # Drain to the freshest queued frame so the stream never lags behind.
            while True:
                try:
                    frame = _frame_queue.get_nowait()
                except queue.Empty:
                    break
            if _SHOW_MASKS and _MASKS_AVAILABLE:
                display = _montage(frame)
            else:
                display = visualize(cv2.resize(frame, _STREAM_SIZE, interpolation=cv2.INTER_AREA))
            ret, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_QUALITY])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except queue.Empty:
            blank = np.zeros((_STREAM_SIZE[1], _STREAM_SIZE[0], 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for frames...", (110, _STREAM_SIZE[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
            ret, jpeg = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_QUALITY])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
        except Exception as e:
            print(f'[VideoStream] Error: {e}')
            time.sleep(0.05)


@app.route('/')
def index():
    return INDEX_HTML


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with _debug_info_lock:
        return jsonify(dict(_debug_info))


@app.route('/command', methods=['POST'])
def command():
    return jsonify({'status': 'ok'})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, leds, stop_event

    ap = argparse.ArgumentParser(description='Project Follower Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT FOLLOWER SERVER — REAL HARDWARE')
    print('=' * 60)

    print('\n[1/4] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/4] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print('  Wheels: ok')

    print('\n[3/4] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting follower agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, leds, stop_event, _frame_queue, _debug_info_lock, _debug_info),
        daemon=True,
        name='FollowAgentThread',
    ).start()
    print('  agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off(); leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nVideo stream: http://localhost:{web_port}/video')
    print('Press Ctrl+C to stop\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off(); leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
