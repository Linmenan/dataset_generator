import math
import numpy as np

def quintic_coeffs(p0, v0, a0, pf, vf, af, sf) -> np.ndarray:
    """
    求解五次多项式 d(s) = a0 + a1 s + ... + a5 s^5 的系数，使
        d(0)=p0, d'(0)=v0, d''(0)=a0,
        d(sf)=pf, d'(sf)=vf, d''(sf)=af
    """
    A = np.array([
        [0,     0,     0,    0,    0,   1],
        [0,     0,     0,    0,    1,   0],
        [0,     0,     0,    2,    0,   0],
        [sf**5, sf**4, sf**3, sf**2, sf, 1],
        [5*sf**4, 4*sf**3, 3*sf**2, 2*sf, 1, 0],
        [20*sf**3,12*sf**2, 6*sf,   2,    0, 0]
    ], dtype=float)
    b = np.array([p0, v0, a0, pf, vf, af], dtype=float)
    return np.linalg.solve(A, b)     # 长度 6

def d_eval(coeffs: np.ndarray, s: float, der: int = 0):
    """五次多项式及其导数"""
    if der == 0:
        return ((coeffs[0]*s**5 + coeffs[1]*s**4 + coeffs[2]*s**3 +
                 coeffs[3]*s**2 + coeffs[4]*s + coeffs[5]))
    elif der == 1:
        return (5*coeffs[0]*s**4 + 4*coeffs[1]*s**3 + 3*coeffs[2]*s**2 +
                2*coeffs[3]*s + coeffs[4])
    elif der == 2:
        return (20*coeffs[0]*s**3 + 12*coeffs[1]*s**2 +
                6*coeffs[2]*s + 2*coeffs[3])
    else:
        raise ValueError("der must be 0/1/2")