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

import tasks.project_lead.packages.agent as agent

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Convoy — Lead</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
  header { padding:12px 16px; background:#1a1a1a; border-bottom:1px solid #333; }
  main { display:flex; justify-content:center; padding:16px; }
  img.stream { max-width:100%; height:auto; border:1px solid #333; background:#000; }
</style></head>
<body>
  <header>Convoy — Lead Bot</header>
  <main><img class="stream" src="/video" alt="camera"></main>
</body></html>
"""

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
encoders   = None
stop_event = threading.Event()

_frame_queue = queue.Queue(maxsize=2)
_debug_info_lock = threading.Lock()
_debug_info = {}

# Display stream: downscale + lower JPEG quality so frames are ~4x smaller over
# wifi (cuts latency), and always send the freshest frame (drain the queue).
_STREAM_SIZE = (480, 360)   # (width, height) of the streamed image
_STREAM_QUALITY = 45        # JPEG quality [0..100] for the stream


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
    cv2.putText(display, f"Route: {info.get('route_idx', 0)}", (10, 90), font, 0.5, (0, 255, 0), 1)

    rl = info.get('red_line')
    if rl:
        cv2.putText(display, f"RedLine: w={rl[0]} d={rl[1]}", (10, h - 60),
                    font, 0.5, (0, 0, 255), 1)
    tag_ids = info.get('apriltag_ids', [])
    if tag_ids:
        cv2.putText(display, f"Tags: {tag_ids}", (10, h - 40), font, 0.5, (255, 100, 255), 1)
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
            small = cv2.resize(frame, _STREAM_SIZE, interpolation=cv2.INTER_AREA)
            display = visualize(small)
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
    global camera, wheels, leds, encoders, stop_event

    ap = argparse.ArgumentParser(description='Project Lead Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT LEAD SERVER — REAL HARDWARE')
    print('=' * 60)

    print('\n[1/5] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/5] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print('  Wheels: ok')

    print('\n[3/5] Initializing wheel encoders (optional)...')
    try:
        from duckiebot.encoder_driver.encoder_driver import WheelEncoderPair
        encoders = WheelEncoderPair()
        print('  Encoders: ok')
    except Exception as e:
        print(f'  Encoders: not available ({e}); turns fall back to lane-reacquire + timeout')
        encoders = None

    print('\n[4/5] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[5/5] Starting lead agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, leds, stop_event, _frame_queue, _debug_info_lock, _debug_info),
        kwargs={'encoders': encoders},
        daemon=True,
        name='LeadAgentThread',
    ).start()
    print('  agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off(); leds.release()
            except Exception:
                pass
        if encoders:
            try:
                encoders.shutdown()
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
        if encoders:
            try:
                encoders.shutdown()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
