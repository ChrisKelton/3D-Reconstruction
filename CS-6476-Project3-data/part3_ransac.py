import itertools
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from affine import Affine

from image_matching import MatchedImages
from part2_fundamental_matrix import get_fundamental_matrix, draw_epipolar_lines, get_epipolar_lines
from serial import load_pickle


def calculate_num_ransac_iterations(
    prob_success: float,
    sample_size: int,
    epsilon: float,
) -> int:
    """

    N = log(1 - p) / log(1 - (1 - epsilon)^s)

        s: the number of points from which the model can be instantiated
            > sample_size
        epsilon: the percentage of outliers in the data
            > epsilon
        p: the requested probability of success
            > prob_success

    :param prob_success:
    :param sample_size:
    :param epsilon:
    :return:
    """
    return int(np.log(1 - prob_success) / np.log(1 - ((1 - epsilon)**sample_size)))


def calculate_residuals(F: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
    return np.abs(np.hstack((pt2, 1)) @ F @ np.append(pt1, [1]))


def ransac_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    prob_success: float = 0.99,
    epsilon: float = 0.5,
    n_pts_to_use_per_estimation: int = 9,
    pixel_th: float = 0.1,
    seed: Optional[int] = 42,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    assert len(pts1) == len(pts2), f"Got different number of points between image1 & image2: {len(pts1)} != {len(pts2)}"

    rng = np.random.default_rng(seed)
    iterations = calculate_num_ransac_iterations(
        prob_success=prob_success,
        sample_size=n_pts_to_use_per_estimation,
        epsilon=epsilon,
    )
    print(f"Number of iterations to use for RANSAC Fundamental Matrix Estimation: {iterations}")

    best_F: Optional[np.ndarray] = None
    inliers1: Optional[np.ndarray] = None
    inliers2: Optional[np.ndarray] = None
    best_inliers: int = 0
    iter_cnt: int = 0

    n_pts: int = len(pts1)
    # for iter_idx in tqdm(range(iterations), desc="Finding Viable Fundamental Matrix"):
    for iter_idx in range(iterations):
        idx = rng.choice(n_pts, n_pts_to_use_per_estimation, replace=False)
        holdout_idx = np.delete(np.arange(n_pts), idx)
        F = get_fundamental_matrix(pts1[idx], pts2[idx])

        # if the fundamental matrix is perfect, then every value will equal 0
        a_F_b = ((np.column_stack([pts2[holdout_idx], np.ones((pts2[holdout_idx].shape[0],))]) @ F) * np.column_stack([pts1[holdout_idx], np.ones(pts1[holdout_idx].shape[0],)])).sum(axis=1)
        # a_F_b = np.asarray([calculate_residuals(F, pt1, pt2) for pt1, pt2 in zip(pts1[holdout_idx], pts2[holdout_idx])])
        inlier_ind = np.where(np.abs(a_F_b) <= pixel_th)[0]
        if len(inlier_ind) > best_inliers:
            best_inliers = len(inlier_ind) + n_pts_to_use_per_estimation
            best_F = F
            inliers1 = np.row_stack([pts1[idx], pts1[inlier_ind]])
            inliers2 = np.row_stack([pts2[idx], pts2[inlier_ind]])
            iter_cnt = iter_idx + 1

    print(f"Best iteration at {iter_cnt}")

    return best_F, inliers1, inliers2


def epipolar_geometric_check(
    fundamental_mat: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    distance_th: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(pts1) == len(pts2), f"Got different number of points between image1 & image2: {len(pts1)} != {len(pts2)}"

    def calculate_distance(epipolar_line: np.ndarray, pt: np.ndarray) -> float:
        return np.linalg.norm(fundamental_mat.dot(epipolar_line) - fundamental_mat.dot(np.append(pt, [1])))

    beg_n_pts: int = len(pts1)

    # lines projected into img1 converging to the epipole in img2
    epipolar_lines = get_epipolar_lines(
        fundamental_mat=fundamental_mat,
        img_right=img1,
        pts_left=pts2,
        pts_right=pts1,
    )
    distances = np.asarray([calculate_distance(epipolar_line, pt) for epipolar_line, pt in zip(epipolar_lines, pts1)])
    good_idx = np.where(distances <= distance_th)
    pts1 = pts1[good_idx]
    pts2 = pts2[good_idx]

    # lines projected into img2 converging to the epipole in img1
    epipolar_lines = get_epipolar_lines(
        fundamental_mat=fundamental_mat,
        img_right=img2,
        pts_left=pts1,
        pts_right=pts2,
    )
    distances = np.asarray([calculate_distance(epipolar_line, pt) for epipolar_line, pt in zip(epipolar_lines, pts2)])
    good_idx = np.where(distances <= distance_th)
    pts1 = pts1[good_idx]
    pts2 = pts2[good_idx]

    return pts1, pts2


def convert_homogenous_line_to_point(homo_line: np.ndarray) -> np.ndarray:
    return homo_line[:2] / homo_line[-1]


# def get_epipoles(
#     fundamental_mat: np.ndarray,
#     img1: np.ndarray,
#     img2: np.ndarray,
#     pts1: np.ndarray,
#     pts2: np.ndarray,
#     seed: Optional[int] = 0,
# ) -> tuple[np.ndarray, np.ndarray]:
#     rng = np.random.default_rng(seed)
#
#     def intersection_point_from_homogenous_lines(
#         homo_line0: np.ndarray,
#         homo_line1: np.ndarray,
#         to_pixels: bool = False,
#     ) -> np.ndarray:
#         intersection_line = np.cross(homo_line0, homo_line1)
#         if to_pixels:
#             return intersection_line[:2] / intersection_line[-1]
#         return intersection_line
#
#     def calculate_epipole(epipolar_lines: np.ndarray) -> np.ndarray:
#         epipoles: list[np.ndarray] = []
#         for epipole_lines in itertools.combinations(epipolar_lines, r=2):
#             epipole = intersection_point_from_homogenous_lines(*epipole_lines)
#             if math.isnan(epipole[0]) or math.isnan(epipole[1]):
#                 continue
#             epipoles.append(epipole)
#         return np.mean(np.stack(epipoles), axis=0)
#
#     epipolar_lines_in_img1_converging_to_the_epipole_in_img2 = get_epipolar_lines(
#         fundamental_mat=fundamental_mat,
#         img_right=img1,
#         pts_left=pts2,
#         pts_right=pts1,
#     )
#     randint: list[int] = []
#     while len(randint) < 7:
#         int_ = rng.integers(low=0, high=epipolar_lines_in_img1_converging_to_the_epipole_in_img2.shape[0] - 1)
#         if int_ not in randint:
#             randint.append(int_)
#     epipole_in_img2 = calculate_epipole(epipolar_lines_in_img1_converging_to_the_epipole_in_img2[randint])
#
#     epipolar_lines_in_img2_converging_to_the_epipole_in_img1 = get_epipolar_lines(
#         fundamental_mat=fundamental_mat,
#         img_right=img2,
#         pts_left=pts1,
#         pts_right=pts2,
#     )
#     randint: list[int] = []
#     while len(randint) < 7:
#         int_ = rng.integers(low=0, high=epipolar_lines_in_img2_converging_to_the_epipole_in_img1.shape[0] - 1)
#         if int_ not in randint:
#             randint.append(int_)
#     epipole_in_img1 = calculate_epipole(epipolar_lines_in_img2_converging_to_the_epipole_in_img1[randint])
#
#     return epipole_in_img1, epipole_in_img2


def compute_epipole(fundamental_mat: np.ndarray) -> np.ndarray:
    """
    compute the (right) epipole from a fundamental matrix F. Use with F.T for (left) epipole

    :param fundamental_mat:
    :return:
    """
    return np.linalg.svd(fundamental_mat)[-1][-1]


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


# TODO: verify/fix this method to get projection matrices
def get_projection_matrices(
    fundamental_mat: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # epipole_in_img1, epipole_in_img2 = compute_epipole(
    #     fundamental_mat=fundamental_mat,
    #     img1=img1,
    #     img2=img2,
    #     pts1=pts1,
    #     pts2=pts2,
    #     seed=seed,
    # )
    epipole_in_img2 = compute_epipole(fundamental_mat.T)
    # P = [I | 0]
    P = np.column_stack([np.identity(3), np.zeros((3,))])

    # P' = [[e']_x F + e'v^T | lambda * e']
    # v: np.ndarray = np.ones((3,)) * rng.random(size=(3,))
    # lambda_val: float = 5e-3
    # # epipole_in_img2 = np.append(epipole_in_img2[:2][::-1], [epipole_in_img2[-1]])
    # P_prime = np.column_stack([
    #     np.cross(epipole_in_img2, fundamental_mat) + epipole_in_img2 @ v.T,
    #     lambda_val * epipole_in_img2,
    # ])
    epipole_in_img2 /= epipole_in_img2[-1]
    epipole_in_img2 = np.append(epipole_in_img2[:2][::-1], [epipole_in_img2[-1]])
    Te = skew(epipole_in_img2)
    P_prime = np.vstack((np.dot(Te, fundamental_mat.T).T, epipole_in_img2)).T

    return P, P_prime


def triangulate_world_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    def _get_world_point(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        A_ = np.asarray([
            (pt1[0] * P1[2].T) - P1[0].T,
            (pt1[1] * P1[2].T) - P1[1].T,
            (pt2[0] * P2[2].T) - P2[0].T,
            (pt2[1] * P2[2].T) - P2[1].T,
        ])
        U, D, V = np.linalg.svd(A_)
        return V[-1][:3]

    world_points: list[np.ndarray] = []
    for pt1, pt2 in zip(pts1, pts2):
        world_points.append(_get_world_point(pt1, pt2))

    return np.stack(world_points)


def calculate_inliers_from_reprojections(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    reprojection_distance_th: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    def project_world_point(world_pt: np.ndarray, P: np.ndarray) -> np.ndarray:
        image_point = P @ np.append(world_pt, [1])
        return image_point[:2] / image_point[-1]

    world_points: np.ndarray = triangulate_world_points(pts1, pts2, P1, P2)
    reprojected_pts1: list[np.ndarray] = []
    reprojected_pts2: list[np.ndarray] = []
    for world_point in world_points:
        reprojected_pts1.append(project_world_point(world_point, P1))
        reprojected_pts2.append(project_world_point(world_point, P2))

    reprojected_pts1 = np.stack(reprojected_pts1)
    reprojected_pts2 = np.stack(reprojected_pts2)

    def reprojection_error_2d(base_pts: np.ndarray, reprojected_pts: np.ndarray) -> np.ndarray:
        return np.sqrt(np.einsum('ij, ij->i', base_pts - reprojected_pts, base_pts - reprojected_pts))

    distances1 = reprojection_error_2d(pts1, reprojected_pts1)
    distances2 = reprojection_error_2d(pts2, reprojected_pts2)

    pts1_idx = np.where(distances1 < reprojection_distance_th)[0]
    pts2_idx = np.where(distances2 < reprojection_distance_th)[0]

    inlier_pts_idx = np.intersect1d(pts1_idx, pts2_idx)

    return pts1[inlier_pts_idx], pts2[inlier_pts_idx]


def get_exif_md(img_path: Path) -> bytes:
    return subprocess.run(["exiftool", str(img_path)], stdout=subprocess.PIPE).stdout


def re_search_exif_md(exif_md: str, key: str) -> str:
    key_search = re.search(f"{key}[ ]+:", exif_md)
    return exif_md[key_search.span()[-1]:key_search.span()[-1] + re.search(r"\\n", exif_md[key_search.span()[-1]:]).span()[0]]


def get_pixel_resolution_from_img(img_path: Path) -> tuple[float, float]:
    exif_md = get_exif_md(img_path)
    return get_pixel_resolution_from_md(str(exif_md))


def get_pixel_resolution_from_md(exif_md: str) -> tuple[float, float]:
    megapixels = float(re_search_exif_md(exif_md, "Megapixels")) * 1e6
    image_width = int(re_search_exif_md(exif_md, "Image Width"))
    image_height = int(re_search_exif_md(exif_md, "Image Height"))

    # pixel focal length in height
    fx = megapixels / image_height
    # pixel focal length in width
    fy = megapixels / image_width

    return fx, fy


def estimate_inital_camera_intrinsic_matrix(img_path: Path) -> np.ndarray:
    exif_md = str(get_exif_md(img_path))
    fx, fy = get_pixel_resolution_from_md(exif_md)
    cx = int(re_search_exif_md(exif_md, "Image Height")) // 2
    cy = int(re_search_exif_md(exif_md, "Image Width")) // 2
    s = 1e-6  # axis skew
    return np.asarray([[fx, s, cx], [0, fy, cy], [0, 0, 1]])


def calculate_essential_matrix(fundamental_mat: np.ndarray, intrinsic1: np.ndarray, intrinsic2: np.ndarray) -> np.ndarray:
    return intrinsic2.T @ fundamental_mat @ intrinsic1


def calculate_essential_matrix_from_images(fundamental_mat: np.ndarray, img1_path: Path, img2_path: Path) -> Affine:
    intrinsic1 = estimate_inital_camera_intrinsic_matrix(img1_path)
    intrinsic2 = estimate_inital_camera_intrinsic_matrix(img2_path)
    E = calculate_essential_matrix(fundamental_mat, intrinsic1, intrinsic2)
    # return calculate_essential_matrix(fundamental_mat, intrinsic1, intrinsic2)
    return Affine(
        a=float(E[0, 0]),
        b=float(E[0, 1]),
        c=float(E[0, 2]),
        d=float(E[1, 0]),
        e=float(E[1, 1]),
        f=float(E[1, 2]),
    )


def main():
    imgs_dir: Path = Path("/home/ckelton/repos/misc/3D-Reconstruction/buddha_mini6")
    img_paths: list[Path] = sorted(imgs_dir.glob("*.png"))
    img_path_names: list[str] = [path.stem for path in img_paths]
    matches_pickle_path: Path = Path("/home/ckelton/repos/misc/3D-Reconstruction/buddha_mini6--features/matches.pickle")
    matches: list[MatchedImages] = load_pickle(matches_pickle_path)

    do_draw_epipolar_lines: bool = True
    min_n_inliers: int = 5
    ransac_fundamental_mat_sample_size: int = 9
    n_inliers_to_draw: int = 10
    reprojection_distance_th: float = 0.5
    out_dir: Path = Path(f"out/part3/n_inliers_to_draw-{n_inliers_to_draw if n_inliers_to_draw > 0 else 'all'}")
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)


    fundamental_mats: dict[str, dict[str, np.ndarray]] = {}
    inliers: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for matched_images in matches:
        img1_path = img_paths[img_path_names.index(matched_images.img1_name)]
        img1 = cv2.imread(str(img1_path))[..., ::-1]
        img2_path = img_paths[img_path_names.index(matched_images.img2_name)]
        img2 = cv2.imread(str(img2_path))[..., ::-1]

        print(f"(image1, image2): ({img1_path.stem}, {img2_path.stem})")

        n_pts = len(matched_images.features1)
        F, inliers1, inliers2 = ransac_fundamental_matrix(
            pts1=matched_images.features1,
            pts2=matched_images.features2,
            n_pts_to_use_per_estimation=ransac_fundamental_mat_sample_size,
        )
        print(f"RANSAC: {len(inliers1)} / {n_pts} \t{(len(inliers1) / n_pts) * 100:.2f} %")
        inliers1, inliers2 = epipolar_geometric_check(
            fundamental_mat=F,
            img1=img1,
            img2=img2,
            pts1=inliers1,
            pts2=inliers2,
        )
        print(f"Epipolar Geometric Checks: {len(inliers1)} / {n_pts} \t{(len(inliers1) / n_pts) * 100:.2f}%")
        if len(inliers1) < min_n_inliers:
            print("\n")
            continue

        if len(inliers1) >= ransac_fundamental_mat_sample_size:
            F, inliers1, inliers2 = ransac_fundamental_matrix(
                pts1=inliers1,
                pts2=inliers2,
                n_pts_to_use_per_estimation=ransac_fundamental_mat_sample_size,
            )
            print(f"RANSAC: {len(inliers1)} / {n_pts} \t{(len(inliers1) / n_pts) * 100:.2f} %")

        if len(inliers1) < min_n_inliers:
            continue
        P1, P2 = get_projection_matrices(
            fundamental_mat=F,
            img1=img1,
            img2=img2,
            pts1=inliers1,
            pts2=inliers2,
        )
        # triangulate_world_points(pts1=inliers1, pts2=inliers2, P1=P1, P2=P2)
        inliers1, inliers2 = calculate_inliers_from_reprojections(
            pts1=inliers1,
            pts2=inliers2,
            P1=P1,
            P2=P2,
            reprojection_distance_th=reprojection_distance_th,
        )
        print(f"Reprojection Inliers: {len(inliers1)} / {n_pts} \t{(len(inliers1) / n_pts) * 100:.2f}%")
        print("\n")
        if len(inliers1) < 1:
            continue
        E = calculate_essential_matrix_from_images(F, img1_path, img2_path)
        fundamental_mats.setdefault(matched_images.img1_name, {})[matched_images.img2_name] = F
        inliers.setdefault(matched_images.img1_name, {})[matched_images.img2_name] = (inliers1, inliers2)

        if do_draw_epipolar_lines:
            if n_inliers_to_draw > 0 and n_inliers_to_draw < len(inliers1):
                inliers1 = inliers1[:n_inliers_to_draw]
                inliers2 = inliers2[:n_inliers_to_draw]
            out_path = out_dir / f"{matched_images.img1_name}--{matched_images.img2_name}.png"
            draw_epipolar_lines(
                F=F,
                img_left=img1,
                img_right=img2,
                pts_left=inliers1,
                pts_right=inliers2,
                out_path=out_path,
                show_fig=False,
            )


if __name__ == '__main__':
    main()
