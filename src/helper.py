import numpy as np


def draw_cross(mat):
    [h, w] = mat.shape
    mat_u1 = np.roll(mat, 1, axis=0)
    mat_u2 = np.roll(mat_u1, 1, axis=0)
    mat_d1 = np.roll(mat, -1, axis=0)
    mat_d2 = np.roll(mat_d1, -1, axis=0)
    mat_l1 = np.roll(mat, 1, axis=1)
    mat_l2 = np.roll(mat_l1, 1, axis=1)
    mat_r1 = np.roll(mat, -1, axis=1)
    mat_r2 = np.roll(mat_r1, -1, axis=1)
    return mat + mat_u1 + mat_u2 + mat_d1 + mat_d2 + mat_l1 + mat_l2 + mat_r1 + mat_r2
