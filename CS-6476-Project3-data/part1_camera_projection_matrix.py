from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(csv_filepath: Path) -> pd.DataFrame:
    return pd.read_csv(str(csv_filepath), header=None, index_col=None)


def get_camera_projection_matrix(
    world_points: np.ndarray,
    pixel_points: np.ndarray,
    use_svd: bool = True,
) -> np.ndarray:
    """
    Compute the projection matrix that goes from world 3D coordinates to 2D image coordinates.
    Recall that using homogeneous coordinates the equation for moving from 3D world to 2D camera coordinates is:

    |u|   |u*s|   |m_11   m_12    m_13    m_14|             | X |
    |v| ~ |v*s| = |m_21   m_22    m_23    m_24|     x       | Y |
    |1|   | s |   |m_31   m_32    m_33    m_34|             | Z |
                                                            | 1 |

    Another way of writing this equation is:

                u = (m_11*X + m_12*Y + m_13*Z + m14) / (m_31*X + m_32*Y + m_33*Z + m_34)

            ->  0 = m_11*X + m_12*Y + m_13*Z + m_14 - m_31*u*X - m_32*u*Y - m_33*u*Z - m_34*u

        ------------

                v = (m_21*X + m_22*Y + m_23*Z + m_24) / (m_31*X + m_32*Y + m_33*Z + m_34)

            ->  0 = m_21*X + m_22*Y + m_23*Z + m_24 - m_31*v*X - m_32*v*Y - m_33*v*Z - m_34*v

    Therefore, we solve the following system of equations:

                                                                                | m_11 |
                                                                                | m_12 |
                                                                                | m_13 |
        |X_1 Y_1 Z_1  1   0   0   0   0  -u_1*X_1 -u_1*Y_1 -u_1*Z_1 -u_1|       | m_14 |        | 0 |
        | 0   0   0   0  X_1 Y_1 Z_1  1  -v_1*X_1 -v_1*Y_1 -v_1*Z_1 -v_1|       | m_21 |        | 0 |
        |                          ...                                  |   x   | m_22 |    =   |...|
        |X_n Y_n Z_n  1   0   0   0   0  -u_n*X_n -u_n*Y_n -u_n*Z_n -u_n|       | m_23 |        | 0 |
        | 0   0   0   0  X_n Y_n Z_n  1  -v_n*X_n -v_n*Y_n -v_n*Z_n -v_n|       | m_24 |        | 0 |
                                                                                | m_31 |
                                                                                | m_32 |
                                                                                | m_33 |
                                                                                | m_34 |

    :return:
    """
    if use_svd:
        A = np.zeros((pixel_points.size, 12))
    else:
        A = np.zeros((pixel_points.size, 11))
    b = np.zeros((pixel_points.size, 1))

    even = np.arange(0, A.shape[0] - 1, 2)
    A[even, :3] = world_points
    A[even, 3] = 1
    A[even, 8:11] = -1 * pixel_points[:, 0][..., None] * world_points
    if use_svd:
        A[even, 11] = -1 * pixel_points[:, 0]
    b[even] = pixel_points[:, 0][..., None]

    odd = np.arange(1, A.shape[0], 2)
    A[odd, 4:7] = world_points
    A[odd, 7] = 1
    A[odd, 8:11] = -1 * pixel_points[:, 1][..., None] * world_points
    if use_svd:
        A[odd, 11] = -1 * pixel_points[:, 1]
    b[odd] = pixel_points[:, 1][..., None]

    if use_svd:
        U, S, V = np.linalg.svd(A)
        proj_mat = V[-1].T.reshape(3, 4) * -1
    else:
        ATA = A.T @ A
        ATb = A.T @ b
        M = np.linalg.inv(ATA) @ ATb
        proj_mat = np.append(M.T, [1]).reshape(3, 4)

    return proj_mat


def plot_projections(proj_mat: np.ndarray, world_points: np.ndarray, pixel_points: np.ndarray, out_path: Optional[Path] = None):
    fig = plt.figure(figsize=(7, 7))
    residuals = []
    for world_point, pixel_point in zip(world_points, pixel_points):
        proj_pixel_point = proj_mat @ np.append(world_point, [1])
        proj_pixel_point = (proj_pixel_point / proj_pixel_point[-1])[:2]
        plt.scatter(pixel_point[0], pixel_point[1], c="green")
        plt.scatter(proj_pixel_point[0], proj_pixel_point[1], c="red", marker="+")
        residual = np.linalg.norm(proj_pixel_point - pixel_point)
        residuals.append(residual)
        print(f"Pixel Point (projected, actual): ({proj_pixel_point}, {pixel_point})\n"
              f"Distance: {residual}\n")
    print(f"Average reprojection error: {np.average(residuals)}")
    plt.title(f"Average Reprojection Error: {np.average(residuals):.5f}")
    plt.show()
    if out_path is not None:
        fig.savefig(str(out_path))
    plt.close(fig)


def calculate_camera_center(proj_mat: np.ndarray) -> np.ndarray:
    Q = proj_mat[:, :3]
    m_4 = proj_mat[:, -1]
    camera_center_world_norm = -np.linalg.inv(Q) @ m_4
    print(f"camera center [world]: {camera_center_world_norm}")

    return camera_center_world_norm


def main():
    data_path: Path = Path("data/CCB_GaTech")
    out_path: Path = Path("out/part1")
    out_path.mkdir(exist_ok=True, parents=True)

    pts2d_norm_pic_a = np.asarray(read_csv(data_path / "pts2d-norm-pic_a.txt"))
    pts3d_norm = np.asarray(read_csv(data_path / "pts3d-norm.txt"))

    proj_mat = get_camera_projection_matrix(pts3d_norm, pts2d_norm_pic_a, use_svd=True)
    plot_projections(proj_mat, pts3d_norm, pts2d_norm_pic_a, out_path / "reprojections.png")
    camera_center = calculate_camera_center(proj_mat)


if __name__ == '__main__':
    main()
