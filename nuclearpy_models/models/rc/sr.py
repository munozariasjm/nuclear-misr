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
            0: "K*(0.9502984598011356 + 1.36*(1/Z)) + 0.017042713515038155*(P - (Z/N))*np.exp(S) - S",
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
            # 11: "((((oz * ((2.5112647929755023 / (Z + -0.37892908364382)) * 1.0319078765296386)) ** Z) + S) * -0.0037334035169424708)",
            # 12: "(((((0.7043599688790316 * -0.18018570969789297) + 0.12820228100582828) / (0.23845339742013413 - S)) / 1.0438595661715504) ** Z)",
            # 13: "((0.0016509066361613859 / (on + -9.448405598196786)) * ((P ** 1.3702681472747735) / 1.525287162904831))",
        }
        self.max_index = len(list(self.expr_dict.keys())) - 1
        self.optimal_params_mean = np.array(
            [
                1.00159966,
                1.14549367,
                1.13821116,
                1.45251976,
                1.21183011,
                1.09285672,
                0.95328495,
                -0.08159929,
                1.02284745,
                1.55923727,
            ]
        )
        self.optimal_params_std = np.array(
            [
                3.03947602e-04,
                6.20367988e-02,
                8.10207149e-02,
                4.20530466e-01,
                5.71420112e-01,
                3.74672574e-01,
                6.64226845e-02,
                2.45251729e-01,
                6.75270036e-01,
                7.40305174e-01,
            ]
        )

    def get_expresion_term(self, index):
        return self.expr_dict[index]

    def __call__(self, Z, N, index=-1, bst=True) -> float:
        features = self._get_features(Z, N)
        if index == -1:
            index = self.max_index - 1
        elif index == 0:
            return self.predict_symb_terms(
                features, self.get_expresion_term(0)
            ), np.abs(self.predict_symb_terms(features, self.get_expresion_term(1)))

        terms = [self.get_expresion_term(i) for i in range(0, index)]
        if not terms:
            terms = [self.get_expresion_term(0)]
        bst_output, unc = self.predict_with_uncertainty_boostrapping(Z, N, index)
        if not bst:
            return np.sum(self.predict_symb_terms(features, terms[index])), unc
        else:
            return bst_output, unc

    @staticmethod
    def _get_features(Z, N):
        """Returns a dictionary for the features starting with the given Z and N:
        ['Z', 'A', 'N', 'x', 'ee', 'eo', 'oe', 'oo', 'P', 'ax', 'K', 'S', 'nmz', 'zmn']
        """

        A = Z + N
        ee = int(Z % 2 == 0 and N % 2 == 0)
        eo = int(Z % 2 == 0 and N % 2 != 0)
        oe = int(Z % 2 != 0 and N % 2 == 0)
        oo = int(Z % 2 != 0 and N % 2 != 0)
        d = ee - oo
        x = (N - Z)**2
        P = compute_P(Z, N)
        h  = Z / N
        K = A ** (1/3)
        S = (N - Z) / A
        t = A ** (2/3)

        return {
            "Z": Z,
            "A": A,
            "N": N,
            "x": x,
            "ee": ee,
            "eo": eo,
            "oe": oe,
            "oo": oo,
            "t": t,
            "h": h,
            "P": P,
            "K": K,
            "S": S,
            "d": d,

        }

    @staticmethod
    def predict_symb_terms(features, model_str):
        return eval(model_str, None, features)

    def predict_term(self, Z, N, term):
        feats = self._get_features(Z, N)
        return self.predict_symb_terms(feats, term)

    def predict_with_uncertainty_boostrapping(
        self,
        Z,
        N,
        index,
        N_ITERATIONS=10_000,
    ):
        terms = [self.get_expresion_term(i) for i in range(0, index)]
        preds_per_term = np.array([self.predict_term(Z, N, term) for term in terms])
        preds = np.dot(
            np.random.normal(
                self.optimal_params_mean[: len(terms)],
                self.optimal_params_std[: len(terms)],
                (N_ITERATIONS, len(terms)),
            ),
            preds_per_term,
        )
        extra_term = self.predict_term(Z, N, self.get_expresion_term(index + 1))
        unc = np.max(np.abs(preds - preds.mean(axis=0)), axis=0) + np.abs(extra_term)
        return preds.mean(), unc


sr_rc = SrRc()
