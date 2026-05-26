"""utils.py — 2D vector / angle 헬퍼 (NVIDIA DOPE 원본).

length / dot_product / normalize / determinant / inner_angle / py_ang
"""
from math import acos, pi, sqrt
import numpy as np


def length(v):
    return sqrt(v[0] ** 2 + v[1] ** 2)


def dot_product(v, w):
    return v[0] * w[0] + v[1] * w[1]


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    cosx = dot_product(v, w) / (length(v) * length(w))
    rad = acos(cosx)
    return rad * 180 / pi


def py_ang(A, B=(1, 0)):
    """B 기준 A 의 시계방향 각도 (0~360°)."""
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det < 0:
        return inner
    else:
        return 360 - inner
