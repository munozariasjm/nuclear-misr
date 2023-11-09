####
# SR guided for correction of the binding energy
from typing import Any
import numpy as np
from .semf import semf_be
from .dz_10 import dz_be
from nuclearpy_models.utils.physics import PhysicsQualities
from ..rc.sr import sr_rc

pq = PhysicsQualities()


class SRBEModels:
    def __init__(self, Z=None, N=None, base_model: str = "base") -> None:
        self.features = ["Z", "N", "d", "P", "I", "O", "T"]
        assert base_model in [
            "base",
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
        if isinstance(Z, int) and isinstance(N, int):
            self.Z = Z
            self.N = N
            self.params = self._build_params(Z, N)

    def _build_params(self, Z, N):
        def distance_to_magic(Z, N, u, v, m, n):
            """
            Euclidean distance to the nearest magic number
            """
            near_z = min(u, m)
            near_n = min(v, n)
            return np.sqrt(np.square(N - near_z) + np.square(N - near_n))  # / (Z + N)

        self.I = pq.S(N, Z)
        self.A = N + Z
        self.x = N - Z
        self.P = pq.compute_P(Z, N)
        self.d = pq.compute_d(Z, N)
        self.u = pq.protons_in_shell(Z)
        self.v = pq.neutrons_in_shell(N)
        self.m = pq.protons_for_shell(Z)
        self.n = pq.neutrons_for_shell(N)
        self.o = (self.m * self.n) / (self.A)
        self.w = distance_to_magic(Z, N, self.u, self.v, self.m, self.n)
        self.p2 = pq.p2(Z, N)
        return (
            self.I,
            self.A,
            self.x,
            self.P,
            self.d,
            self.u,
            self.v,
            self.m,
            self.n,
            self.o,
            self.w,
        )

    def even_even_be(self, Z, N):
        return dz_be(Z, N) + self.corr_dz(Z, N)

    def odd_even_be(self, Z, N):
        # Here Z odd N even
        # predict Z-1, N
        ee = self.even_even_be(Z - 1, N)
        # spr = B(Z+1, N) - B(Z, N) => B(Z, N) = B(Z-1, N) + spr
        spr = self.spr(Z - 1, N)
        be = ee + spr
        return be

    def even_odd_be(self, Z, N):
        # Here Z even N odd
        # predict Z, N-1
        ee = self.even_even_be(Z, N - 1)
        # snu = B(Z, N+1) - B(Z, N) => B(Z, N) = B(Z, N-1) + snu(Z, N - 1)
        snu = self.snu(Z, N - 1)
        be = ee + snu
        return be

    def odd_odd_be(self, Z, N):
        # Here Z odd N odd
        # predict Z-1, N-1
        ee = self.even_even_be(Z - 1, N - 1)
        # spr = B(Z+1, N) - B(Z, N) => B(Z, N) = B(Z-1, N) + spr
        spr = self.spr(Z - 1, N - 1)
        # snu = B(Z, N+1) - B(Z, N) => B(Z, N) = B(Z, N-1) + snu
        snu = self.snu(Z - 1, N - 1)
        be = ee + spr + snu
        return be

    def be(self, Z, N):
        if Z % 2 == 0 and N % 2 == 0:
            return self.even_even_be(Z, N)
        elif Z % 2 == 1 and N % 2 == 0:
            return self.odd_even_be(Z, N)
        elif Z % 2 == 0 and N % 2 == 1:
            return self.even_odd_be(Z, N)
        else:
            return self.odd_odd_be(Z, N)

    @classmethod
    def __call__(cls, Z, N) -> Any:
        try:
            nuclei = cls(Z, N)
            return nuclei.be(Z, N)
        except Exception as e:
            print("Error in computing binding energy", (Z, N))
            print(e)
            return None

    #################################################
    #
    #  SR

    def spr(self, Z, N):
        """
        Computes the proton separation energy
        spr = B(Z+1, N) - B(Z, N)
        """
        I, A, x, P, d, u, v, m, n, o, w = self._build_params(Z, N)
        return (
            (
                (
                    (
                        (((x / 0.7078585905714976) - -2.841942869152135) / w)
                        * (
                            (
                                22.78277908524047
                                - (
                                    (
                                        ((v * 2.037985266612706) / Z)
                                        + -0.29373476921233693
                                    )
                                    ** n
                                )
                            )
                            + (
                                (((x + v) - Z) / 22.78277908524047)
                                * (
                                    (1.6199262914668977 + 1.6708891006560465)
                                    / 0.687652816457333
                                )
                            )
                        )
                    )
                    - ((Z + (v**1.1029085076424152)) / 22.78277908524047)
                )
                - (-0.2446069658849346 / 0.319346771685812)
            )
            * (
                (((w + -2.841942869152135) + -0.6045269200855017) / N)
                + 0.1576408521155684
            )
        ) + (
            (((u / I) ** (0.36351764953728877 + o)) * 1.5126532467258116)
            ** 1.5126532467258116
        )

    def snu(self, Z, N):
        """
        Computes the neutron separation energy
        snu = B(Z, N+1) - B(Z, N)
        """
        I, A, x, P, d, u, v, m, n, o, w = self._build_params(Z, N)
        return (
            (Z * ((20.329616986072715 / N) - -0.024103610855896913))
            - 10.319743646975327
        ) - (
            (
                -0.042368976888709234
                * (
                    v
                    - (
                        (
                            ((x / 10.319743646975327) + -1.3097301309547276)
                            - 1.5220714814035015
                        )
                        * (
                            (
                                (((v / 0.519504017910704) + x) / 10.319743646975327)
                                - ((10.319743646975327 / 0.10468022010473134) / N)
                            )
                            + (
                                (
                                    (
                                        (
                                            ((-21.5207003038607 / N) ** N)
                                            - -0.024103610855896913
                                        )
                                        * ((n * (v / N)) ** v)
                                    )
                                    + 0.19388412915885597
                                )
                                ** -1.3097301309547276
                            )
                        )
                    )
                )
            )
            + (10.319743646975327 / N)
        )

    def corr_dz(self, Z, N):
        """
        Correction to the Duflo-Zuker model
        """
        I, A, x, P, d, u, v, m, n, o, w = self._build_params(Z, N)
        P2 =
        def inv_sq(x, y):
            return x / (y**2)

        # return (
        #     (((inv_sq(Z, 1.5361079196045446) - x) / Z) * (((A - u) + m) / Z))
        #     * (
        #         0.16473240914372989
        #         - (
        #             (
        #                 (0.547539847235243**u)
        #                 - (((v**0.3676539463635918) / A) * (v - x))
        #             )
        #             - (
        #                 (
        #                     (
        #                         ((0.04715064103019326 / w) ** w)
        #                         ** (0.8452294323915344**w)
        #                     )
        #                     ** Z
        #                 )
        #                 * 1.6171051008896653
        #             )
        #         )
        #     )
        # ) + -0.11447030587325317
        sr0 = (
            (
                (
                    (
                        A
                        * (
                            (np.square(Z) * (-0.03576613587611979 / I))
                            - -10.053125126715656
                        )
                    )
                    - (31.34713980458172 + P2)
                )
                + (I + 1)
            )
            - (
                np.square(np.square(np.square(x) * 0.00046013632080631493))
                * 1.0441931308557395
            )
        ) + (
            ((((np.square(x) - 1) * -5.629386686969966) / Z) - 23.31313801160723)
            + -0.1561051166310651
        )

        sr0 += (
            (
                (
                    (
                        (127171.90534554968 / A**3)
                        + (
                            (0.6734075939914219 ** (0.6734075939914219**Z))
                            ** (x**x)
                        )
                    )
                    + ((0.6734075939914219 ** (0.6734075939914219**Z)) ** (x**n))
                )
                - (-(0.628618881357727**x))
            )
            + ((0.8688859562375671 ** (m + v)) * (Z * 0.09026654692458154))
        ) - 1.0601093059117397
        sr0 += (
            (
                ((v * (0.22482729292393602**o)) / Z)
                * (
                    (
                        (n - (w / 2.9776111309827695**3))
                        * ((x / Z) - 0.117497917834406)
                    )
                    - (o ** (u**0.6169971543941742))
                )
            )
            + (((v ** (P * 0.11678524038301441)) / A) + (0.5846603044846019**x))
        ) * -2.297849315710175
