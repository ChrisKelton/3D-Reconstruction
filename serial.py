import copyreg
import pickle
from pathlib import Path
from typing import Any

import cv2


def _pickle_keypoints(point: cv2.KeyPoint):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


def _pickle_matches(match: cv2.DMatch):
    return cv2.DMatch, (match.queryIdx, match.trainIdx, match.imgIdx, match.distance)


copyreg.pickle(cv2.DMatch().__class__, _pickle_matches)


def save_pickle(obj: Any, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj
