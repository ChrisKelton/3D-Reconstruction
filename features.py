__all__ = [
    "extract_features",
    "extract_keypoints",
    "plot_keypoints",
    "draw_bruteforce_matches",
    "draw_bruteforce_ratio_test_matches",
    "get_matches",
]

from pathlib import Path
from typing import Optional, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from serial import save_pickle


def _extract_keypoints(img: np.ndarray, keypoint_detector: Optional[Callable] = None) -> tuple[cv2.KeyPoint, ...]:
    if keypoint_detector is None:
        keypoint_detector = cv2.SIFT_create()
    return keypoint_detector.detect(img, None)


def extract_keypoints(img_paths: list[Path], pickle_path: Path):
    kps: list[tuple[cv2.KeyPoint, ...]] = []
    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        kp = _extract_keypoints(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY))
        kps.append(kp)
    save_pickle(tuple(kps), pickle_path)


def _extract_features(img: np.ndarray, feature_detector: Optional[Callable] = None):
    if feature_detector is None:
        feature_detector = cv2.SIFT_create()
    return feature_detector.detectAndCompute(img, None)


def extract_features(img_paths: list[Path], pickle_path: Path):
    features: list[tuple[tuple[cv2.KeyPoint, ...], np.ndarray]] = []
    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        feature = _extract_features(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY))
        features.append(feature)
    save_pickle(tuple(features), pickle_path)


def plot_keypoints(img_path: Path, kp: tuple[cv2.KeyPoint, ...], out_path: Path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.drawKeypoints(gray, kp, img)
    cv2.imwrite(str(out_path), img)


def _get_bruteforce_matches(desc1: np.ndarray, desc2: np.ndarray) -> list[cv2.DMatch]:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


def _get_bruteforce_knn_matches(desc1: np.ndarray, desc2: np.ndarray, k: int = 2) -> tuple[tuple[cv2.DMatch, ...], ...]:
    bf = cv2.BFMatcher()
    return bf.knnMatch(desc1, desc2, k=k)


def get_matches(
    desc1: np.ndarray,
    desc2: np.ndarray,
    k: int = 2,
    img1_name: Optional[str] = None,
    img2_name: Optional[str] = None,
) -> list[cv2.DMatch]:
    print(f"Finding matches for: {img1_name}, {img2_name}")
    matches = _get_bruteforce_knn_matches(desc1, desc2, k)
    n_initial_matches = len(matches)
    print(f"Initial matches found: {n_initial_matches}")
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(f"Matches passed Lowe's Ratio: {len(good)} / {n_initial_matches}\t{(len(good) / n_initial_matches) * 100:.2f}%\n")

    return good


def draw_bruteforce_matches(
    img1: Path,
    img2: Path,
    features1: tuple[tuple[cv2.KeyPoint, ...], np.ndarray],
    features2: tuple[tuple[cv2.KeyPoint, ...], np.ndarray],
    n_matches_to_draw: int = 10,
):
    matches = _get_bruteforce_matches(features1[1].astype(np.uint8), features2[1].astype(np.uint8))
    img1 = cv2.imread(str(img1))
    img2 = cv2.imread(str(img2))
    img_matched = cv2.drawMatches(img1, features1[0], img2, features2[0], matches[:n_matches_to_draw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matched)
    plt.show()
    plt.close()


def draw_bruteforce_ratio_test_matches(
    img1: Path,
    img2: Path,
    features1: tuple[tuple[cv2.KeyPoint, ...], np.ndarray],
    features2: tuple[tuple[cv2.KeyPoint, ...], np.ndarray],
):
    matches = get_matches(features1[1].astype(np.uint8), features2[1].astype(np.uint8))
    img1 = cv2.imread(str(img1))
    img2 = cv2.imread(str(img2))
    img_matched = cv2.drawMatchesKnn(img1, features1[0], img2, features2[0],matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matched)
    plt.show()
    plt.close()
