__all__ = ["show_image", "draw_epipolar_lines"]
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(img: Path):
    img = cv2.imread(str(img))
    plt.imshow(img[..., ::-1])
    plt.show()
    plt.close()


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
