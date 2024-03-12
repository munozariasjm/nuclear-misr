from nuclearpy_models.utils.physics import PhysicsQualities
import numpy as np
import sys

pq = PhysicsQualities()
"""
["K*(0.9502984598011356 + 1.258334808536715*(1/Z)) + 0.017042713515038155*(P - (Z/N))*np.exp(S) - S"
(np.square(0.06607824565977677) * (d - (((((((np.square(P) - np.exp(0.8557356856877293)) / h) * np.exp(3.267293299752022)) + A) / np.square(Z)) * (np.exp(3.267293299752022) - np.square(P ** 1.0436541692574126))) + 1.6967299664570696)))
'(((on - 3.2638621674211596) + oz) * (-6.212472578529901e-5 * (oz - ((on - oz) / 0.8149545777451943))))',
 '((((h ** N) - P) + 1.6075150804926084) / -954.0159520953583)',
 '(((0.002003867687626223 / (oz + -0.46273537077375637)) / (-0.48550082610955214 + P)) * (P ** 1.9787077730657177))',
 '((-4.9489047702697845e-6 / (0.9420409700915601 - h)) / 0.09525830755304454)',
 '((((N / (A - 57.42169913174474)) / 41.245571253453825) / (0.13154516140090133 + on)) / 41.245571253453825)',
 '((((-0.00011816308572832208 / (0.00022320017496487048 - on)) ** N) * (oe - -0.5075098534022552)) / 0.0001272063986793674)',
 '((((-0.6164943592049766 + (((oz / h) * -0.019535405196981315) - -0.019535405196981315)) ** N) / 0.6109634910349819) * oo)',
 '((-0.0004231223865735906 - (((on ** oo) * (-0.0052107021784933705 * on)) / A)) / h)',
 '(((0.001489590152797498 / (1.162580718921562 - oz)) * (((oe * h) + eo) - P)) / 0.33113516554290906)',
 '((((oz * ((2.5112647929755023 / (Z + -0.37892908364382)) * 1.0319078765296386)) ** Z) + S) * -0.0037334035169424708)',
 '(((((0.7043599688790316 * -0.18018570969789297) + 0.12820228100582828) / (0.23845339742013413 - S)) / 1.0438595661715504) ** Z)',
 '((0.0016509066361613859 / (on + -9.448405598196786)) * ((P ** 1.3702681472747735) / 1.525287162904831))']
"""


# get the path to this file
class SrRc:
    def __init__(self):

        self.expr_dict = {
            0: "K*(0.9502984598011356 + 1.34*(1/Z)) + 0.017042713515038155*(P - (Z/N))*np.exp(S) - S",
            1: "(np.square(0.06607824565977677) * (d - (((((((np.square(P) - np.exp(0.8557356856877293)) / h) * np.exp(3.267293299752022)) + A) / np.square(Z)) * (np.exp(3.267293299752022) - np.square(P ** 1.0436541692574126))) + 1.6967299664570696)))",
            2: "(((on - 3.2638621674211596) + oz) * (-6.212472578529901e-5 * (oz - ((on - oz) / 0.8149545777451943))))",
            3: "((((h ** N) - P) + 1.6075150804926084) / -954.0159520953583)",
            4: "(((0.002003867687626223 / (oz + -0.46273537077375637)) / (-0.48550082610955214 + P)) * (P ** 1.9787077730657177))",
            5: "((-4.9489047702697845e-6 / (0.9420409700915601 - h)) / 0.09525830755304454)",
            6: "((((N / (A - 57.42169913174474)) / 41.245571253453825) / (0.13154516140090133 + on)) / 41.245571253453825)",
            7: "((((-0.00011816308572832208 / (0.00022320017496487048 - on)) ** N) * (oe - -0.5075098534022552)) / 0.0001272063986793674)",
            8: "((((-0.6164943592049766 + (((oz / h) * -0.019535405196981315) - -0.019535405196981315)) ** N) / 0.6109634910349819) * oo)",
            9: "((-0.0004231223865735906 - (((on ** oo) * (-0.0052107021784933705 * on)) / A)) / h)",
            10: "(((0.001489590152797498 / (1.162580718921562 - oz)) * (((oe * h) + eo) - P)) / 0.33113516554290906)",
            11: "((((oz * ((2.5112647929755023 / (Z + -0.37892908364382)) * 1.0319078765296386)) ** Z) + S) * -0.0037334035169424708)",
            12: "(((((0.7043599688790316 * -0.18018570969789297) + 0.12820228100582828) / (0.23845339742013413 - S)) / 1.0438595661715504) ** Z)",
            13: "((0.0016509066361613859 / (on + -9.448405598196786)) * ((P ** 1.3702681472747735) / 1.525287162904831))",
        }
        self.max_index = len(list(self.expr_dict.keys())) - 1

    def firs_order_bayesian(self, Z, N):
        pass  # TODO

    def __call__(self, Z, N, index=0) -> float:
        features = self._get_features(Z, N)
        if index == -1:
            index = len(self.expr_dict) - 1
        if index == 0:
            complete_model = self.expr_dict[0]
        else:
            complete_model = " + ".join([self.expr_dict[i] for i in range(0, index)])
        symbonic_terms_pred = self.predict_symb_terms(features, complete_model)
        return symbonic_terms_pred

    def predict_single_term(self, Z, N, index):
        features = self._get_features(Z, N)
        return self.predict_symb_terms(features, self.expr_dict[index])

    @staticmethod
    def _get_features(Z, N):
        """Returns a dictionary for the features starting with the given Z and N:
        ['Z', 'A', 'N', 'x', 'ee', 'eo', 'oe', 'oo', 'P', 'ax', 'K', 'S', 'nmz', 'zmn']
        """
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]
        # νp(n) is the difference between the proton (neutron) number
        # of a particular nucleus and the nearest magic number.
        clossest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
        clossest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
        vp = abs(Z - clossest_z)
        vn = abs(N - clossest_n)
        P = (vp * vn) / (vp + vn + 1e-6)
        A = Z + N
        ee = int(Z % 2 == 0 and N % 2 == 0)
        eo = int(Z % 2 == 0 and N % 2 != 0)
        oe = int(Z % 2 != 0 and N % 2 == 0)
        oo = int(Z % 2 != 0 and N % 2 != 0)
        d = ee - oo
        x = (N - Z) ** 2
        h = Z / N
        K = A ** (1 / 3)
        S = (N - Z) / A
        # Number of protons over the closest magic number from below
        oz = Z - min(z_magic_numbers, key=lambda x: abs(x - Z))
        # Number of neutrons over the closest magic number from below
        on = N - min(n_magic_numbers, key=lambda x: abs(x - N))
        return {
            "Z": Z,
            "A": A,
            "N": N,
            "x": x,
            "ee": ee,
            "eo": eo,
            "oe": oe,
            "oo": oo,
            "h": h,
            "P": P,
            "K": K,
            "S": S,
            "d": d,
            "oz": oz,
            "on": on,
        }

    @staticmethod
    def predict_symb_terms(features, model_str):
        return eval(model_str, None, features)


sr_rc = SrRc()
