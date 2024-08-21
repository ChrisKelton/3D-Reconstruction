__all__ = ["MatchingSettings", "cameras_run", "CameraPair"]
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from image_matching import MatchedImages
from inliers import epipolar_geometric_check, calculate_inliers_from_reprojections
from utils import draw_epipolar_lines


@dataclass
class MatchingSettings:
    min_n_inliers: int = 5
    ransac_fundamental_mat_sample_size: int = 9
    n_inliers_to_draw: int = 10
    reprojection_distance_th: float = 0.5
    do_draw_epipolar_lines: bool = True


def get_fundamental_matrix(img1_pts: np.ndarray, img2_pts: np.ndarray) -> np.ndarray:
    """
    The definition of the fundamental matrix is:

                        | f_11 f_12 f_13 |      | u |
        |u' v' 1|   x   | f_21 f_22 f_23 |  x   | v |   =   0
                        | f_31 f_32 f_33 |      | 1 |

    for a point (u, v, 1) in image A, and a point (u', v', 1) in image B. See
    https://faculty.cc.gatech.edu/~hays/compvision2021/proj3/proj3.pdf Appendix A for the full derivation.
    Note: the fundamental matrix is sometimes defined as the transpose of the above matrix with the left and right image
    points swapped.

    Another way of writing these matrix equations is:

                        | f_11*u f_12*v f_13 |
        |u' v' 1|   x   | f_21*u f_22*v f_23 |  =   0.
                        | f_31*u f_32*v f_33 |

    Which is the same as:

        (f_11*u*u' + f_12*v*u' + f_13*u' + f_21*u*v' + f_22*v*v' + f_23*v' + f_31*u + f_32*v + f_33) = 0.

    Given corresponding points you get one equation per point pair. With 8 or more points you can solve this.

    The least squares estimate of F is full rank; however, a proper fundamental matrix is a rank 2. As such we must
    reduce its rank. In order to do this, we can decompose F using singular value decomposition into the matrices
    U*Sigma*V' = F. We can then construct a rank 2 matrix by setting the smallest singular value in Sigma to zero thus
    generating Sigma_2. The fundamental matrix is then easily calculated as F = U*Sigma_2*V'.

    :param img1_pts:
    :param img2_pts:
    :return:
    """
    assert len(img1_pts) == len(img2_pts), f"Got different number of points between image1 & image2: {len(img1_pts)} != {len(img2_pts)}"

    A = np.zeros((len(img1_pts), 8))
    A[:, 0] = img1_pts[:, 0] * img2_pts[:, 0]
    A[:, 1] = img1_pts[:, 1] * img2_pts[:, 0]
    A[:, 2] = img2_pts[:, 0]
    A[:, 3] = img1_pts[:, 0] * img2_pts[:, 1]
    A[:, 4] = img1_pts[:, 1] * img2_pts[:, 1]
    A[:, 5] = img2_pts[:, 1]
    A[:, 6] = img1_pts[:, 0]
    A[:, 7] = img1_pts[:, 1]
    b = -np.ones((len(img1_pts), 1))

    F = np.linalg.inv(A.T @ A) @ (A.T @ b)
    F = np.concatenate((F.ravel(), [1]), axis=0).reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[np.argmin(S)] = 0
    F = (U @ np.diag(S)) @ V

    return F


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


def compute_epipole(fundamental_mat: np.ndarray) -> np.ndarray:
    """
    compute the (right) epipole from a fundamental matrix F. Use with F.T for (left) epipole

    :param fundamental_mat:
    :return:
    """
    return np.linalg.svd(fundamental_mat)[-1][-1]


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


# def skew(x):
#     return np.array([[0, -x[0], x[1]], [x[0], 0, -x[2]], [-x[1], x[2], 0]])


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
    # img1 = left image
    epipole_in_img1 = compute_epipole(fundamental_mat.T)
    # P = [I | 0]
    P = np.column_stack([np.identity(3), np.zeros((3,))])

    # P' = [[e']_x F + e'v^T | lambda * e']
    v: np.ndarray = np.ones((3,)) * rng.random(size=(3,))
    lambda_val: float = 5e-3
    # epipole_in_img2 = np.append(epipole_in_img2[:2][::-1], [epipole_in_img2[-1]])
    # P_prime = np.column_stack([
    #     np.cross(epipole_in_img1, fundamental_mat) + epipole_in_img1 @ v.T,
    #     lambda_val * epipole_in_img1,
    # ])
    P_prime = np.column_stack([np.cross(epipole_in_img1, fundamental_mat), epipole_in_img1])
    # P_prime = np.column_stack([Te @ fundamental_mat, epipole_in_img1])

    # epipole_in_img1 /= epipole_in_img1[-1]
    # epipole_in_img1 = np.append(epipole_in_img1[:2][::-1], [epipole_in_img1[-1]])
    # Te = skew(epipole_in_img1)
    # P_prime = np.vstack((np.dot(Te, fundamental_mat.T).T, epipole_in_img1)).T

    return P, P_prime


def get_projection_matrix(fundamental_mat: np.ndarray) -> np.ndarray:
    epipole_in_view1 = compute_epipole(fundamental_mat.T)
    epipole_in_view1 /= epipole_in_view1[-1]
    epipole_in_view1 = np.append(epipole_in_view1[:2][::-1], [epipole_in_view1[-1]])
    Te = skew(epipole_in_view1)
    return np.vstack((np.dot(Te, fundamental_mat.T).T, epipole_in_view1)).T


@dataclass
class CameraPair:
    img1: Path
    img2: Path
    F: np.ndarray
    inliers1: np.ndarray
    inliers2: np.ndarray


def cameras_run(
    matched_images: MatchedImages,
    matching_settings: Optional[MatchingSettings] = None,
    debug_out_dir: Optional[Path] = None,
) -> Optional[CameraPair]:
    if matching_settings is None:
        matching_settings = MatchingSettings()

    if debug_out_dir is None:
        matching_settings.do_draw_epipolar_lines = False

    img1_path = matched_images.img1_path
    img1 = cv2.imread(str(img1_path))[..., ::-1]
    img2_path = matched_images.img2_path
    img2 = cv2.imread(str(img2_path))[..., ::-1]

    print(f"(image1, image2): ({img1_path.stem}, {img2_path.stem})")

    n_pts = len(matched_images.features1)
    F, inliers1, inliers2 = ransac_fundamental_matrix(
        pts1=matched_images.features1,
        pts2=matched_images.features2,
        n_pts_to_use_per_estimation=matching_settings.ransac_fundamental_mat_sample_size,
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
    if len(inliers1) < matching_settings.min_n_inliers:
        print("\n")
        return None

    if len(inliers1) >= matching_settings.ransac_fundamental_mat_sample_size:
        F, inliers1, inliers2 = ransac_fundamental_matrix(
            pts1=inliers1,
            pts2=inliers2,
            n_pts_to_use_per_estimation=matching_settings.ransac_fundamental_mat_sample_size,
        )
        print(f"RANSAC: {len(inliers1)} / {n_pts} \t{(len(inliers1) / n_pts) * 100:.2f} %")

    if len(inliers1) < matching_settings.min_n_inliers:
        return None

    P1, P2 = get_projection_matrices(
        fundamental_mat=F,
        img1=img1,
        img2=img2,
        pts1=inliers1,
        pts2=inliers2,
    )
    # P1 = get_projection_matrix(fundamental_mat=F)
    # P2 = get_projection_matrix(fundamental_mat=F.T)
    inliers1, inliers2 = calculate_inliers_from_reprojections(
        pts1=inliers1,
        pts2=inliers2,
        P1=P1,
        P2=P2,
        reprojection_distance_th=matching_settings.reprojection_distance_th,
    )
    print(f"Reprojection Inliers: {len(inliers1)} / {n_pts} \t{(len(inliers1) / n_pts) * 100:.2f}%")
    print("\n")
    if len(inliers1) < matching_settings.min_n_inliers:
        return None

    if matching_settings.do_draw_epipolar_lines:
        if matching_settings.n_inliers_to_draw > 0 and matching_settings.n_inliers_to_draw < len(inliers1):
            inliers1 = inliers1[:matching_settings.n_inliers_to_draw]
            inliers2 = inliers2[:matching_settings.n_inliers_to_draw]
        out_path = debug_out_dir / f"{matched_images.img1_name}--{matched_images.img2_name}.png"
        draw_epipolar_lines(
            F=F,
            img_left=img1,
            img_right=img2,
            pts_left=inliers1,
            pts_right=inliers2,
            out_path=out_path,
            show_fig=False,
        )

    return CameraPair(
        img1=img1_path,
        img2=img2_path,
        F=F,
        inliers1=inliers1,
        inliers2=inliers2,
    )
