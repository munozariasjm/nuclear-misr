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
        else:
            self.Z = None
            self.N = None

        self.sr_expresion = "(((np.square(((I / -0.21632577578508072) / 0.3327977461148198) - -1.718648387419444) + np.log(K)) - (P / 1.1909381123436593)) * (np.log(K) ** 1.2625650864325817)) + ((np.square((0.2553620423031214 - (S * 0.5781059994609957)) * K) ** K) - ((((1.2623901619069073 ** S) - I) ** ax) * 1.2623901619069073)) + (0.0003175414464184741 * (np.square((((0.89221991337525 - P) * x) + x) + x) - (1.270452957835343 ** x))) + ((((np.exp(-2.0996728851985353 * h) ** (0.005631256387370711 ** P)) - 0.05832814612747451) * P) - ((np.square(0.5833553810143978) ** P) ** np.square(nmz))) + ((np.square(0.08706447746840168 ** (P ** K)) * (np.square(P + P) + (I ** I))) ** (1.1152266348570135 - -1.7805646495152845)) + ((((((1.062080115295015 / Z) ** (P ** N)) * np.square(P)) ** rc) * (-0.6163885719951542 - oe)) * np.exp(P)) + ((K - np.exp(1.41485592297657)) * np.exp(1.41485592297657 - ((P * P) + ((nmz / 2.5996293351796678) * (ax * I))))) + ((((np.square(S) + -0.0569565775325078) * (np.exp(1.1551483397501359 - S) * ((2.824609919559794 + P) - np.square(P)))) * K) * S) + ((((((0.9167180950314712 ** np.square(nmz)) * np.square(ax)) / 0.7059297794884732) / np.square(Z)) * (0.9167180950314712 - -0.7989481840117367)) * (P - 0.9167180950314712)) + (-5.079219166362556e-5 * (np.exp((I - (((P ** K) + oo) * 0.05553169741673529)) * ax) - np.exp(P * 1.5135270933122495))) + (((((((K / np.exp(P)) ** np.exp(d)) ** np.square(P)) + oo) - 1.1452183772154203) * S) + 0.06680761021169167) + ((0.8149351110501565 ** A) * (((((ax * -0.9508228637524836) * d) - x) * np.exp(oe)) * np.exp(oo + K))) + (((0.4187750432149856 ** (P + ax)) ** (P + ax)) * ((d + np.square(P - ax)) + (ax + d))) + ((((I * np.square(0.7767019451749606)) / np.exp(ax)) * P) * (np.exp(P) - np.exp(rc + (0.41560651666888704 - oo)))) + (0.6003182656560041 * ((((-0.9437386457825158 + P) ** d) * np.square(I)) - (0.2787595743882821 * -0.13268520508357776))) + ((np.exp((-0.7030161085447372 * (P ** P)) + 0.08232241870274408) * np.square((P - 0.2448185969750887) * (oo - oe))) - 0.0707359708866333) + ((((np.exp(-0.06280499026247874) ** np.square(A)) ** (2.7127733878799575e-13 ** nmz)) - np.exp(-0.16243557426994634)) * 0.5006680041449113) + (((((np.log(h) ** d) * ((-1.2669935467922135 * np.log(h)) * np.square(0.1115882302888174))) ** np.square(nmz)) / 2.524515680915015) / 0.7254616053892528) + ((x - nmz) * np.square(-0.08873178397684205))"

    @staticmethod
    def even_even(Z, N):
        if Z % 2 == 0 and N % 2 == 0:
            return 1
        else:
            return 0

    @staticmethod
    def even_odd(Z, N):
        if Z % 2 == 0 and N % 2 != 0:
            return 1
        else:
            return 0

    @staticmethod
    def odd_even(Z, N):
        if Z % 2 != 0 and N % 2 == 0:
            return 1
        else:
            return 0

    @staticmethod
    def odd_odd(Z, N):
        if Z % 2 != 0 and N % 2 != 0:
            return 1
        else:
            return 0

    @staticmethod
    def compute_P(Z, N):
        """promiscuity factor"""
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 114]
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]
        # νp(n) is the difference between the proton (neutron) number
        # of a particular nucleus and the nearest magic number.
        clossest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
        clossest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
        vp = abs(Z - clossest_z)
        vn = abs(N - clossest_n)
        return (vp * vn) / (vp + vn + 1e-6)

    def get_rc_features(self, Z, N):
        """Returns a dictionary for the features starting with the given Z and N:
        ['Z', 'A', 'N', 'x', 'ee', 'eo', 'oe', 'oo', 'P', 'ax', 'K', 'S', 'nmz', 'zmn']
        """
        A = Z + N
        ee = int(Z % 2 == 0 and N % 2 == 0)
        eo = int(Z % 2 == 0 and N % 2 != 0)
        oe = int(Z % 2 != 0 and N % 2 == 0)
        oo = int(Z % 2 != 0 and N % 2 != 0)
        x = N - Z
        P = self.compute_P(Z, N)
        ax = np.abs(x)
        K = A ** (1 / 3)
        S = x / A
        I = ax / A
        nmz = N % Z
        zmn = Z % N
        return {
            "Z": Z,
            "A": A,
            "N": N,
            "x": x,
            "ee": ee,
            "eo": eo,
            "oe": oe,
            "oo": oo,
            "P": P,
            "ax": ax,
            "K": K,
            "S": S,
            "I": I,
            "nmz": nmz,
            "zmn": zmn,
        }

    def predict_rc(self, Z, N):
        rc = "((((K * 0.7786429249127564) + 0.71804292995309) - (S * np.exp(-0.46213789724155524))) + (((Z + np.exp(P * 0.6308208753666963)) + (P + ee)) * 0.0037326576210471424)) + ((0.6994662418783195 ** Z) * ((np.square(K) - np.exp(P)) - ((((1.434798749113789 ** (Z - P)) ** 0.6816672938606969) * ((ax - oe) - ee)) * 0.02577560527777368))) + ((np.square(0.13228290187880484) * (((((S / 1.27025099449954) - 0.13228290187880484) * K) * np.exp(-0.28961804989733414)) + (np.square(-0.28961804989733414) ** nmz)))) + ((np.square(2.1985278043239423e-5) * (P - np.exp(P * (3.061974013739838 - (((ee + eo) * S) * (ax + -1.0639061411973467))))))) + ((((np.square((0.00011587460212412281 ** A) ** (0.00011587460212412281 ** P)) * ax) + np.log(0.3177579358277494)) * np.square(-0.030143217193425133 + -0.0038958947532610295)))"
        values = self.get_rc_features(Z, N)
        return eval(rc, None, values)

    def get_rc(self, Z, N):
        return self.predict_rc(Z, N)

    def get_features(self, Z, N):
        """Returns a dictionary for the features starting with the given Z and N:
        ['Z', 'A', 'N', 'x', 'ee', 'eo', 'oe', 'oo', 'P', 'ax', 'K', 'S', 'nmz', 'zmn']
        """
        A = Z + N
        ee = int(Z % 2 == 0 and N % 2 == 0)
        eo = int(Z % 2 == 0 and N % 2 != 0)
        oe = int(Z % 2 != 0 and N % 2 == 0)
        oo = int(Z % 2 != 0 and N % 2 != 0)
        x = N - Z
        P = self.compute_P(Z, N)
        ax = np.abs(x)
        K = A ** (1 / 3)
        S = x / A
        I = ax / A
        nmz = N % Z
        zmn = Z % N
        rc = self.get_rc(Z, N)
        # from nuclearpy_models.models.rc import mnp_rc
        # rc = mnp_rc(Z, N)
        h = Z * (Z - 1)
        d = ee - oo
        return {
            "Z": Z,
            "A": A,
            "N": N,
            "x": x,
            "ee": ee,
            "eo": eo,
            "oe": oe,
            "oo": oo,
            "P": P,
            "ax": ax,
            "K": K,
            "S": S,
            "I": I,
            "nmz": nmz,
            "zmn": zmn,
            "rc": rc,
            "h": h,
            "d": d,
        }

    def predict(self, Z, N):
        values = self.get_features(Z, N)
        return eval(self.sr_expresion, None, values) + semf_be(Z, N)

    def __call__(self, Z, N):
        return (self.predict(Z, N) + dz_be(Z, N)) / 2
