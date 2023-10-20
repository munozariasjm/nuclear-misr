####
# SR guided for correction of the binding energy
import numpy as np
from .semf import semf_be
from .dz_10 import dz_be


class SRBEModels:
    def __init__(self, base_model: str) -> None:
        self.features = ["Z", "N", "d", "P", "I", "O", "T"]
        assert base_model in [
            "dz",
            "semf",
            "ens",
        ], """
        Base model must be one of the following:
        - `dz` Duflo-Zuker model
        - `semf` Semi-empirical mass formula
        - `ens` Ensemble of the two models
        """
        self.base = base_model
        self.params = []

    def __call__(self, Z, N):
        return self.eval(Z, N)

    def __repr__(self) -> str:
        return f"SRBEModels()"

    def eval(self, Z, N):
        A = Z + N
        d = self.compute_d(Z, N)
        P = self.compute_P(Z, N)
        I = self.compute_I(Z, N)
        O = A ** (2 / 3)
        T = abs(Z - N)
        K = A ** (-1 / 3)
        v = self.protons_in_shell(Z)
        u = self.neutrons_in_shell(N)

        if "dz" in self.base.lower():
            return self.sr_dz_be(Z, N, K, A, d, P, I, O, T, u, v)
        elif "semf" in self.base.lower():
            return self.sr_semf_be(Z, N, d, P, I, O, T)
        elif "ens" in self.base.lower():
            return self.sr_ens_be(Z, N, d, P, I, O, T)

    @staticmethod
    def compute_P(Z, N):
        """promiscuity factor"""
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]
        clossest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
        clossest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
        vp = abs(Z - clossest_z)
        vn = abs(N - clossest_n)
        return (vp * vn) / (vp + vn + 1e-6)

    @staticmethod
    def compute_d(Z, N):
        """d is the difference between the number of protons and the number of neutrons"""
        if (Z % 2 == 0) and (N % 2 == 0):
            return 1
        elif (Z % 2 == 1) and (N % 2 == 1):
            return -1
        else:
            return 0

    @staticmethod
    def compute_I(Z, N):
        """I is the isospin"""
        return (N - Z) / (Z + N)

    def semf_corr(
        self,
        Z,
        N,
        d,
        P,
        I,
        O,
        T,
        params=[
            2.15,
            0.618,
            0.523,
            0.364,
            1.01,
            0.0708,
            2.11,
            0.57,
            0.469,
            2.94,
            0.102,
            1.17,
            1.12,
        ],
    ):
        """semf corrected"""
        corr = (
            -O * (2.15 * 0.618**P - 0.523) * (O - 0.364)
            - 1.01
            * (0.0708 * O - 2.11)
            * (
                ((N + O) * (N - O**0.57 + O)) ** 0.469 * (2.94 - 0.102 * O)
                + (1.17 - (N * O + 1.12) ** I) * (O - 0.364)
            )
            * (N**2) ** I
        ) / (O * (I - 0.469) * (O - 0.364))
        return np.float64(corr)

    def dz_corr(
        self,
        Z,
        N,
        K,
        A,
        d,
        P,
        I,
        O,
        T,
        u,
        v
        # params=[8.59e-01, 1.24e00, 5.37e-08, 0.00083, 1.96e-18],
    ):
        """dz corrected"""
        val = (
            (((P + P) / N) - I)
            * (
                (P * 0.18641840277500055)
                - (
                    -1.2335152270631435
                    - (
                        (
                            (((1.0443901604611923**v) - (N - Z)) - (P - (u - v)))
                            * ((u + (((1.0443901604611923**u) / Z) ** d)) - N)
                        )
                        / A
                    )
                )
            )
        ) + 0.12817605758700557
        return np.float64(val)

    def sr_semf_be(self, Z, N, d, P, I, O, T):
        return self.semf_corr(Z, N, d, P, I, O, T) + semf_be(Z, N)

    def sr_dz_be(self, Z, N, K, A, d, P, I, O, T, u, v):
        return self.dz_corr(Z, N, K, A, d, P, I, O, T, u, v) + dz_be(Z, N)

    def sr_ens_be(self, Z, N, d, P, I, O, T, weights=[0.5, 0.5]):
        """
        Ensemble of the two models
        """
        return (
            weights[0] * self.sr_semf_be(Z, N, d, P, I, O, T)
            + weights[1] * self.sr_dz_be(Z, N, K, A, d, P, I, O, T, u, v)
        ) / sum(weights)

    @staticmethod
    def protons_in_shell(Z):
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        return min(z_magic_numbers, key=lambda x: abs(x - Z))

    @staticmethod
    def neutrons_in_shell(N):
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]
        return min(n_magic_numbers, key=lambda x: abs(x - N))
