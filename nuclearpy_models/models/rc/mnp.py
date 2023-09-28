import numpy as np

###
# Implementation of the Isospin dependent NP model
# described in: https://link.springer.com/article/10.1140/epja/i2015-15040-1


def compute_P(Z, N):
    """promiscuity factor"""
    z_magic_numbers = [8, 20, 28, 50, 82, 126]
    n_magic_numbers = [8, 20, 28, 50, 82, 126, 184]
    # νp(n) is the difference between the proton (neutron) number
    # of a particular nucleus and the nearest magic number.
    clossest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
    clossest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
    vp = abs(Z - clossest_z)
    vn = abs(N - clossest_n)
    return (vp * vn) / (vp + vn + 1e-6)


def mnp_rc(Z: int, N: int):
    """modified_np_isospin"""

    A = N + Z
    Z = int(Z)
    N = int(N)
    P = compute_P(Z, N)
    ro = 1.2321
    a = 0.1534
    b = 1.3358
    c = 0.4317
    dd = 0.1225
    if N % 2 == 0 and Z % 2 == 0:  # even-even
        delta = 1
    elif N % 2 == 1 and Z % 2 == 1:  # odd-odd
        delta = -1
    else:
        delta = 0

    factor = 1 + (-(a * (N - Z) / A) + (b / A) + (c * P / A) + (dd * delta / A))
    return ro * (A ** (1 / 3)) * factor * np.sqrt(3 / 5)
