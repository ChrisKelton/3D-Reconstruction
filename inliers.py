__all__ = [
    "get_epipolar_lines",
    "epipolar_geometric_check",
    "triangulate_world_points",
    "calculate_inliers_from_reprojections",
]
import numpy as np


def get_epipolar_lines(
    fundamental_mat: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    to_cartesian: bool = False,
) -> np.ndarray:
    H, W = img_right.shape[:2]
    # corner points, as homogenous (x, y, 1)
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([W, 0, 1])
    p_bl = np.asarray([W, H, 1])
    p_br = np.asarray([W, H, 1])

    # the equation of the line through two points can be determined by taking the 'cross product' of their
    # homogenous coordinates

    # left and right border lines, for the right image
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    line_points: list[np.ndarray] = []

    for p in pts_left:
        p = np.hstack((p, 1))[:, np.newaxis]
        # get defn of epipolar line in right image, corresponding to left point p
        l_e = np.dot(fundamental_mat, p).squeeze()

        if to_cartesian:
            # find where epipolar line in right image crosses the left and image borders
            p_l = np.cross(l_e, l_l)
            p_r = np.cross(l_e, l_r)
            # convert back from homogenous to cartesian by dividing by 3rd entry
            # draw line between one point on left border, and on the right border
            x = [p_l[0] / p_l[2], p_r[0] / p_r[2]]
            y = [p_l[1] / p_l[2], p_r[1] / p_r[2]]

            line_points.extend([x, y])
        else:
            line_points.append(l_e)

    return np.asarray(line_points)


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
