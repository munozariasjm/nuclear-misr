import numpy as np
import pandas as pd


class SRModel:
    def __init__(self):
        pass

    def sr_target(self, N, Z):
        params = [
            0.98643564,
            1.00640007,
            4.80168938,
            0.02922448,
            0.28229105,
            1.02390465,
            0.99105636,
            -0.18597759,
        ]
        P = self._compute_P(Z, N)
        delta = self._compute_delta(Z, N)
        A = Z + N
        K = A ** (1 / 3)
        fact1 = K * (params[0])
        fact2 = params[1] * (I * P) / (Z * K)
        fact3 = params[2] / Z
        fact4 = (params[3]) ** (params[4] * P**2) / Z
        fact5 = params[5] * P / Z
        fact6 = params[6] * I * (1 + delta - Z) / Z
        return fact1 + fact2 + fact3 + fact4 + fact5 + fact6 + params[7]

    @staticmethod
    def _compute_P(Z, N):
        """promiscuity factor"""
        z_magic_numbers = [8, 20, 28, 50, 82, 126]
        n_magic_numbers = [8, 20, 28, 50, 82, 126, 184]
        # νp(n) is the difference between the proton (neutron) number
        # of a particular nucleus and the nearest magic number.
        closest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
        closest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
        vp = abs(Z - closest_z)
        vn = abs(N - closest_n)
        return (vp * vn) / (vp + vn + 1e-6)

    @staticmethod
    def _compute_delta(Z, N):
        """d is the difference between the number of protons and the number of neutrons"""
        if (Z % 2 == 0) and (N % 2 == 0):
            return 1
        elif (Z % 2 == 1) and (N % 2 == 1):
            return -1
        else:
            return 0
