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
        T = Z - N
        if "dz" in self.base.lower():
            return self.sr_dz_be(Z, N, d, P, I, O, T)
        elif "semf" in self.base.lower():
            return self.sr_semf_be(Z, N, d, P, I, O, T)
        elif "ens" in self.base.lower():
            return self.sr_ens_be(Z, N, d, P, I, O, T)

    @staticmethod
    def compute_P(Z, N):
        """promiscuity factor"""
        z_magic_numbers = [8, 20, 28, 50, 82, 126]
        n_magic_numbers = [8, 20, 28, 50, 82, 126, 184]
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
            -O * (params[0] * params[1] ** P - params[2]) * (O - params[3])
            - params[4]
            * (params[5] * O - params[6])
            * (
                ((N + O) * (N - O ** params[7] + O)) ** params[8]
                * (params[9] - params[10] * O)
                + (params[11] - (N * O + params[12]) ** I) * (O - params[3])
            )
            * (N**2) ** I
        ) / (O * (I - params[8]) * (O - params[3]))
        return np.float64(corr)

    def dz_corr(
        self,
        Z,
        N,
        d,
        P,
        I,
        O,
        T,
        params=[
            2.32,
            1.15,
            0.0523,
            2.82,
            2.01e-5,
            1.01,
            3.53,
            2.56e-5,
            0.73,
            0.0558,
            19.1,
            0.0219,
            0.0446,
            1.49,
        ],
    ):
        """dz corrected"""

        val = (
            -(
                Z * I
                + params[0]
                * (P - params[1])
                * (
                    -Z
                    * (
                        params[2]
                        * P
                        * (
                            O ** params[3]
                            * (
                                params[4] * I * (-params[5] * P + I + params[6])
                                + params[7]
                            )
                            + params[8]
                        )
                        - params[9]
                    )
                    + P
                    + params[10] * (P + I) * (params[11] * O + params[12]) ** O
                    + params[13]
                )
            )
            / Z
        )
        return np.float64(val)

    def sr_semf_be(self, Z, N, d, P, I, O, T):
        return self.semf_corr(Z, N, d, P, I, O, T) + semf_be(Z, N)

    def sr_dz_be(self, Z, N, d, P, I, O, T):
        return self.dz_corr(Z, N, d, P, I, O, T) + dz_be(Z, N)

    def sr_ens_be(self, Z, N, d, P, I, O, T, weights=[0.5, 0.5]):
        """
        Ensemble of the two models
        """
        return (
            weights[0] * self.sr_semf_be(Z, N, d, P, I, O, T)
            + weights[1] * self.sr_dz_be(Z, N, d, P, I, O, T)
        ) / sum(weights)
