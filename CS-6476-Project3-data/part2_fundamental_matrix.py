from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from part1_camera_projection_matrix import read_csv


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
    # https://danielwedge.com/fmatrix/
    # perform rank deprivation, so the epipolar lines & epipoles coincide
    S[np.argmin(S)] = 0
    F = (U @ np.diag(S)) @ V

    return F


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


def draw_epipolar_lines(
    F: np.ndarray,
    img_left: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    out_path: Optional[Path] = None,
    show_fig: bool = True,
):
    # -------------- lines in the RIGHT image --------------------
    H1, W1 = img_right.shape[:2]
    # corner points, as homogeneous (x, y, 1)
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([W1, 0, 1])
    p_bl = np.asarray([0, H1, 1])
    p_br = np.asarray([W1, H1, 1])

    # the equation of the line through two points can be determined by taking the
    # 'cross product' of their homogeneous coordinates.

    # left and right border lines, for the right image
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

    ax[1].imshow(img_right)
    ax[1].autoscale(False)
    ax[1].scatter(pts_right[:, 0], pts_right[:, 1], marker="o", s=20, c="yellow", edgecolor="red")

    # epipolar_lines = get_epipolar_lines(
    #     fundamental_mat=F,
    #     img_right=img_right,
    #     pts_left=pts_left,
    #     pts_right=pts_right,
    #     to_cartesian=True,
    # )
    # for idx in range(len(epipolar_lines) // 2):
    #     idx *= 2
    #     x, y = epipolar_lines[idx:idx + 2]
    #     ax[1].plot(x, y, linewidth=1, c="blue")

    for p in pts_left:
        p = np.hstack((p, 1))[:, np.newaxis]
        # get defn of epipolar line in right image, corresponding to left point p
        l_e = np.dot(F, p).squeeze()
        # find where epipolar line in right image crosses the left and image borders
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        # convert back from homogeneous to cartesian by dividing by 3rd entry
        # draw line between one point on left border, and on the right border
        x = [p_l[0] / p_l[2], p_r[0] / p_r[2]]
        y = [p_l[1] / p_l[2], p_r[1] / p_r[2]]
        ax[1].plot(x, y, linewidth=1, c="blue")

    # ------------ lines in the LEFT image --------------------
    imgh_left, imgw_left = img_left.shape[:2]

    # corner points, as homogeneous (x,y,1)
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([imgw_left, 0, 1])
    p_bl = np.asarray([0, imgh_left, 1])
    p_br = np.asarray([imgw_left, imgh_left, 1])

    # left and right border lines, for left image
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    ax[0].imshow(img_left)
    ax[0].autoscale(False)
    ax[0].scatter(
        pts_left[:, 0], pts_left[:, 1], marker="o", s=20, c="yellow", edgecolors="red"
    )
    for p in pts_right:
        p = np.hstack((p, 1))[:, np.newaxis]
        # defn of epipolar line in the left image, corresponding to point p in the right image
        l_e = np.dot(F.T, p).squeeze()
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0] / p_l[2], p_r[0] / p_r[2]]
        y = [p_l[1] / p_l[2], p_r[1] / p_r[2]]
        ax[0].plot(x, y, linewidth=1, c="blue")

    if out_path is not None:
        plt.savefig(str(out_path))
    if show_fig:
        plt.show()
    plt.close(fig)


def main():
    data_path: Path = Path("data/CCB_GaTech")
    out_path: Path = Path("out/part2")
    out_path.mkdir(exist_ok=True, parents=True)

    pts2d_pic_a = np.asarray(read_csv(data_path / "pts2d-pic_a.txt"))
    pic_a = cv2.imread(str(data_path / "pic_a.jpg"))
    pts2d_pic_b = np.asarray(read_csv(data_path / "pts2d-pic_b.txt"))
    pic_b = cv2.imread(str(data_path / "pic_b.jpg"))

    F = get_fundamental_matrix(pts2d_pic_a, pts2d_pic_b)
    draw_epipolar_lines(F, pic_a[..., ::-1], pic_b[..., ::-1], pts2d_pic_a, pts2d_pic_b, out_path / "epipolar-lines.png")


if __name__ == '__main__':
    main()
