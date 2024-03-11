from nuclearpy_models.utils.physics import PhysicsQualities
import numpy as np
import sys

pq = PhysicsQualities()
"""

['K*(0.950 + 1.25833*(1/Z)) + 0.01704*(P - (Z/N))*np.exp(S) - S",0.015
 '((((((-4.170148008453756 + P) * (np.square(P) - 1.3253142984692627)) * np.square(1.1458945325796521)) / np.square(Z)) + -0.0063204725787710915) + -0.0063204725787710915)',
 '((((Z + x) * np.exp(3.3879551024085384 - (Z - d))) - (0.002479916805146936 * oo)) - (0.002479916805146936 * oo))',
 '((np.exp((-54.05849098893126 + (Z + ((P ** 1.3684295182204842) / 1.3684295182204842))) - 0.10512557716285173) + -0.0019202662672411484) + (-0.0019202662672411484 * (np.square(oe * 0.06737083847130876) ** P)))',
 '((((np.exp(0.28467286615027493) + np.exp(0.4420576509305732)) - np.exp(P * 0.4420576509305732)) / np.square(A)) * (np.exp((1.8966891916349986 - x) + 1.1889288266392157) - np.exp(1.4820120904542367)))',
 '((np.square((K ** (-239.71854366553032 - 0.8641485864234607)) / -239.71854366553032) ** (K ** -2.7661447487490607)) * ((-239.71854366553032 * P) + -239.71854366553032))',
 '(((np.square(S * h) * np.exp(h)) * ((np.square(h - eo) * P) + np.square(0.8236398869052595))) / (Z + -27.454836176969))',
 '((((np.exp((0.01828027605613407 * A) - 0.5904736179751292) - np.exp(1.0538078960245933)) * np.square(0.018401515379972165)) * (h - P)) * (np.exp(1.7100407868559488 + oo) + 1.0078868528728977))',
 '((np.square(0.004461434544688601) * (np.square((0.004461434544688601 / (-0.3566049981534501 - oo)) * x) + np.exp(oe))) / (h + -0.9425541296870317))',
 '((((((P / (P ** P)) ** K) / K) * h) / Z) * d)',
 '((x - np.square(P ** P)) * (0.4698220964534311 ** (Z + d)))',
 '(np.square(0.001476317795081674) / (0.23715367684165248 - S))',
 'np.square((1.3347448721067152e-5 * N) * ((-0.700341167230781 / (-0.0022345606810030374 + S)) - (((N ** oo) + -0.700341167230781) / -1.0062971148001005)))']
"""


# get the path to this file
class SrRc:
    def __init__(self):

        self.expr_dict = {
            0: "K*(0.950 + 1.25833*(1/Z)) + 0.01704*(P - (Z/N))*np.exp(S) - S",
            1: "(np.square(0.06607824565977677) * (d - (((((((np.square(P) - np.exp(0.8557356856877293)) / h) * np.exp(3.267293299752022)) + A) / np.square(Z)) * (np.exp(3.267293299752022) - np.square(P ** 1.0436541692574126))) + 1.6967299664570696)))",
            # 2: "((((Z + x) * np.exp(3.3879551024085384 - (Z - d))) - (0.002479916805146936 * oo)) - (0.002479916805146936 * oo))",
            # 3: "((np.exp((-54.05849098893126 + (Z + ((P ** 1.3684295182204842) / 1.3684295182204842))) - 0.10512557716285173) + -0.0019202662672411484) + (-0.0019202662672411484 * (np.square(oe * 0.06737083847130876) ** P)))",
            # 4: "((((np.exp(0.28467286615027493) + np.exp(0.4420576509305732)) - np.exp(P * 0.4420576509305732)) / np.square(A)) * (np.exp((1.8966891916349986 - x) + 1.1889288266392157) - np.exp(1.4820120904542367)))",
            # 5: "((np.square((K ** (-239.71854366553032 - 0.8641485864234607)) / -239.71854366553032) ** (K ** -2.7661447487490607)) * ((-239.71854366553032 * P) + -239.71854366553032))",
            # 6: "(((np.square(S * h) * np.exp(h)) * ((np.square(h - eo) * P) + np.square(0.8236398869052595))) / (Z + -27.454836176969))",
            # 7: "((((np.exp((0.01828027605613407 * A) - 0.5904736179751292) - np.exp(1.0538078960245933)) * np.square(0.018401515379972165)) * (h - P)) * (np.exp(1.7100407868559488 + oo) + 1.0078868528728977))",
            # 8: "((np.square(0.004461434544688601) * (np.square((0.004461434544688601 / (-0.3566049981534501 - oo)) * x) + np.exp(oe))) / (h + -0.9425541296870317))",
            # 9: "((((((P / (P ** P)) ** K) / K) * h) / Z) * d)",
            # 10: "((x - np.square(P ** P)) * (0.4698220964534311 ** (Z + d)))",
            # 11: "(np.square(0.001476317795081674) / (0.23715367684165248 - S))",
            # 12: "np.square((1.3347448721067152e-5 * N) * ((-0.700341167230781 / (-0.0022345606810030374 + S)) - (((N ** oo) + -0.700341167230781) / -1.0062971148001005)))",
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
        }

    @staticmethod
    def predict_symb_terms(features, model_str):
        return eval(model_str, None, features)


sr_rc = SrRc()
