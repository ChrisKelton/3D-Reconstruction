import shutil
from pathlib import Path

from cameras import MatchingSettings, cameras_run, CameraPair
from image_matching import image_matching, MatchedImages
from serial import load_pickle, save_pickle


def main():
    src_dir: Path = Path("buddha_mini6")
    out_dir: Path = Path("buddha_mini6--features")
    out_dir.mkdir(exist_ok=True, parents=True)
    debug_out_dir: Path = out_dir / "debug"
    overwrite: bool = True

    matching_settings: MatchingSettings = MatchingSettings(
        min_n_inliers=5,
        reprojection_distance_th=500,  # pixels
        n_inliers_to_draw=-1,
    )

    features_pickle_path, matches_pickle_path = image_matching(
        src_dir=src_dir,
        out_dir=out_dir,
        overwrite=overwrite,
    )

    cameras_pickle_path: Path = out_dir / "cameras.pickle"

    if not cameras_pickle_path.exists() or overwrite:
        if debug_out_dir.exists():
            shutil.rmtree(str(debug_out_dir))
        debug_out_dir.mkdir(exist_ok=True, parents=True)

        matches: list[MatchedImages] = load_pickle(matches_pickle_path)
        cameras: list[CameraPair] = []
        for matched_images in matches:
            camera = cameras_run(
                matched_images=matched_images,
                matching_settings=matching_settings,
                debug_out_dir=debug_out_dir,
            )
            if camera is not None:
                cameras.append(camera)

        save_pickle(cameras, cameras_pickle_path)
    else:
        cameras = load_pickle(cameras_pickle_path)


if __name__ == '__main__':
    main()
