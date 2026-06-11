import logging
import re
import threading
import time
import cv2


def update_yaml_values(path, updates):
    """Rewrite top-level scalar keys in a YAML file in place, preserving all
    comments and layout (yaml.dump would destroy the hand-written comments in
    the config files). Keys not found are appended at the end."""
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    for key, value in updates.items():
        sval = f"{value:g}" if isinstance(value, float) else str(value)
        pattern = re.compile(rf"^({re.escape(key)}\s*:\s*)([^#\n]*?)(\s*#.*)?$", re.M)
        if pattern.search(text):
            text = pattern.sub(lambda m: m.group(1) + sval + (m.group(3) or ""), text, count=1)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"{key}: {sval}\n"

    with open(path, "w") as f:
        f.write(text)


class LatestFrame:
    """Single-slot frame holder: the newest frame always wins, never blocks
    and never fills up. Queue-compatible put_nowait so agents can treat it
    like the real servers' frame queue."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put_nowait(self, frame):
        with self._lock:
            self._frame = frame

    def get_latest(self):
        with self._lock:
            return self._frame


class _HttpErrorsOnly(logging.Filter):
    """Pass werkzeug request lines only when status code >= 400."""
    _STATUS_RE = re.compile(r'" (\d{3}) ')

    def filter(self, record):
        m = self._STATUS_RE.search(record.getMessage())
        if m:
            return int(m.group(1)) >= 400
        return True  # non-request lines (startup, errors) always shown


def suppress_http_logs():
    """Call once at server startup to hide 2xx/3xx request noise."""
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)
    log.addFilter(_HttpErrorsOnly())


def make_frame_generator(get_camera, visualize, quality=70, rgb=True):
    """Return an MJPEG generator. rgb=True calls read_rgb(), False calls read()."""
    def generate():
        while True:
            try:
                cam = get_camera()
                if cam is None:
                    time.sleep(0.05)
                    continue

                ok, frame = cam.read_rgb() if rgb else cam.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                display = visualize(frame)
                ret, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + jpeg.tobytes() + b'\r\n')

            except Exception as e:
                print(f'[VideoStream] Error: {e}')
                time.sleep(0.05)

    return generate


def shutdown_cleanup(wheels, camera, stop_event):
    """Stop motors, stop camera, set stop_event."""
    stop_event.set()

    if wheels:
        try:
            print("Stopping motors...")
            wheels.set_wheels_speed(0, 0)
            time.sleep(0.1)
            wheels.set_wheels_speed(0, 0)
        except Exception as e:
            print(f"  Error: {e}")

    if camera:
        try:
            print("Stopping camera...")
            camera.stop()
        except Exception as e:
            print(f"  Error: {e}")

    print("\nShutdown complete!")
