__all__ = ["MatchedImages", "image_matching"]
import itertools
import os
from dataclasses import dataclass
from math import comb
from pathlib import Path

import cv2
import numpy as np

from features import extract_features, get_matches
from serial import save_pickle, load_pickle


@dataclass
class MatchedImages:
    img1_name: str
    img1_path: Path
    img2_name: str
    img2_path: Path
    features1: np.ndarray
    features2: np.ndarray
    matches: list[cv2.DMatch]


def image_matching(
    src_dir: Path,
    out_dir: Path,
    n_matches: int = 10,
    required_n_matches: float = 0.25,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    out_dir.mkdir(exist_ok=True, parents=True)

    img_paths = sorted(src_dir.glob("*.png"))
    features_pickle_path = out_dir / "features.pickle"
    matches_pickle_path = out_dir / "matches.pickle"

    if not features_pickle_path.exists() or overwrite:
        if features_pickle_path.exists():
            os.remove(str(features_pickle_path))
        extract_features(img_paths, features_pickle_path)
        if matches_pickle_path.exists():
            os.remove(str(matches_pickle_path))

    if not matches_pickle_path.exists():
        features = load_pickle(features_pickle_path)

        idx_combinations = itertools.combinations(range(0, len(img_paths)), r=2)
        n_combinations = comb(len(img_paths), 2)
        matched_images: list[MatchedImages] = []
        for idx, (idx1, idx2) in enumerate(idx_combinations):
            print(f"[{idx + 1} / {n_combinations}]")
            img1_name = img_paths[idx1].stem
            img2_name = img_paths[idx2].stem
            features1 = features[idx1]
            features2 = features[idx2]
            matches = get_matches(
                desc1=features1[1],
                desc2=features2[1],
                img1_name=img1_name,
                img2_name=img2_name,
            )
            if len(matches) >= n_matches:
                pts_1: list[np.ndarray] = []
                pts_2: list[np.ndarray] = []
                for m in matches:
                    pts_1.append(features1[0][m.queryIdx].pt)
                    pts_2.append(features2[0][m.trainIdx].pt)
                matched_images.append(
                    MatchedImages(
                        img1_name=img1_name,
                        img1_path=img_paths[idx1],
                        img2_name=img2_name,
                        img2_path=img_paths[idx2],
                        features1=np.asarray(pts_1),
                        features2=np.asarray(pts_2),
                        matches=matches,
                    )
                )

        n_successful_matches = len(matched_images) / n_combinations
        if n_successful_matches < required_n_matches:
            raise RuntimeError(f"Not enough matches. Got {n_successful_matches * 100:.2f}% successful image pairs.")
        else:
            print(f"Got {n_successful_matches * 100:.2f}% successful image pairs.")
        save_pickle(tuple(matched_images), matches_pickle_path)
    else:
        matched_images = load_pickle(matches_pickle_path)

    return features_pickle_path, matches_pickle_path


def main():
    src_dir: Path = Path("buddha_mini6")
    out_dir: Path = Path("buddha_mini6--features")
    out_dir.mkdir(exist_ok=True, parents=True)
    n_matches_to_draw: int = 10
    overwrite: bool = False

    img_paths = sorted(src_dir.glob("*.png"))
    features_pickle_path, keypoints_pickle_path = image_matching(
        src_dir=src_dir,
        out_dir=out_dir,
        overwrite=False,
    )

    # if not keypoints_pickle_path.exists():
    #     # generate keypoints
    #     extract_keypoints(img_paths, keypoints_pickle_path)
    # if not features_pickle_path.exists():
    #     # generate keypoints & descriptors
    #     extract_features(img_paths, features_pickle_path)
    #
    # # # plot keypoints
    # # with open(keypoints_pickle_path, "rb") as f:
    # #     kps = pickle.load(f)
    # # for img_path, kp in zip(img_paths, kps):
    # #     plot_keypoints(img_path, kp, out_dir / img_path.name)
    #
    # # plot keypoints from features
    # with open(features_pickle_path, "rb") as f:
    #     features = pickle.load(f)
    # for img_path, feature in zip(img_paths, features):
    #     plot_keypoints(img_path, feature[0], out_dir / img_path.name)
    #
    # for idx1, idx2 in itertools.combinations(range(0, len(img_paths)), r=2):
    #     draw_bruteforce_matches(
    #         img1=img_paths[idx1],
    #         img2=img_paths[idx2],
    #         features1=features[idx1],
    #         features2=features[idx2],
    #         n_matches_to_draw=n_matches_to_draw,
    #     )
    #     draw_bruteforce_ratio_test_matches(
    #         img1=img_paths[idx1],
    #         img2=img_paths[idx2],
    #         features1=features[idx1],
    #         features2=features[idx2],
    #     )


if __name__ == '__main__':
    main()
