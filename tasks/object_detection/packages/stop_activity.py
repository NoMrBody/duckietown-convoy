from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    raise NotImplementedError("Implement should_stop in stop_activity.py")
