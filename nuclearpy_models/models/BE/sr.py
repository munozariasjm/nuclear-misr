####
# SR guided for correction of the binding energy
from typing import Any, List
import numpy as np


class SrBe:
    def __init__(self, base_model: str = "base") -> None:
        self.exprs_list = [
            "(((Z + ((-1.1518949008311261 / h) * 0.9191416779446455)) + h) * (((32.42652864512878 - (K / h)) * S) - -16.68481023317041))",
            "(((((np.exp(1.1006874673013554) - ((K - 1.3719238061914094) - (S / 0.4559834159964229))) * ((Z - (-2.0129653786217006 / -0.149121978667067)) - 1.1006874673013554)) * ((np.log(K) - (S / 0.32947021405273286)) * 1.1264781107199664)) + (d - P)) + 0.3012755947167558)",
            "((((0.8667226017107358 ** x) * np.exp(1.0957980468921584)) - (np.exp(0.7030577891026748 - (((0.036946866702474115 * Z) ** Z) - ((-0.39995415163224035 * P) * ee))) - S)) + (((-0.12360962447332499 * P) / 0.2905011442378596) * np.log(0.036946866702474115 * Z)))",
            "((((np.log(h ** S) * t) * (np.exp(Z ** 1.1017534523017007) / np.exp(t))) / np.exp(Z)) - (0.29005199356401196 * (((np.exp((t - N) + K) * np.log(0.11228275004501485)) - ee) - 0.8482750079357783)))",
            "((((x * P) + N) * (P + (1.538872526208315e-5 * x))) * ((1.538872526208315e-5 * t) ** P))",
            "((oe / N) - np.exp(((((P ** K) - (P ** N)) * ((P + P) + 0.215478490301157)) - ((h - -0.4261022127969257) + eo)) * (K + (-1.20622071093187 - h))))",
            "((S * 1.3513717643571443) * ((((-1.7835130140476037 - h) * (0.3243294855565521 - S)) * ((np.log(0.8949329729059017) * (np.exp(P) + A)) + (np.exp(1.4593173482869688) / h))) + ((S * 1.3513717643571443) - P)))",
            "((((0.8010663851875898 ** N) * (A + ((A / h) - x))) - 0.111962577802992) * ((((P / np.exp(0.5618598224805517)) - S) - (0.8010663851875898 ** oe)) - S))",
            "((-9.81179689889166e-16 * (((2.0946679715754835 + 0.879009069930303) - P) - oo)) * (((x + -1334.5332266527023) * ((x + (x + -1334.5332266527023)) + np.log(0.07629937107106072))) * ((x - (x + -1334.5332266527023)) ** (K + -2.070055596300072))))",
            "np.exp(-0.8948709195716154 * (((((0.9270832840257225 ** (N + N)) * N) * ((0.9270832840257225 ** N) * N)) + (0.02994986947801994 * x)) + ((((P ** 1.9412112817066696) * (h ** t)) / 0.46865314891483706) - 1.2343874050146584)))",
            "((-0.4788113460721299 + ((((0.012167385837240422 - ((-2.047639966100053 / h) - S)) + (K * K)) * ((P * P) ** (K * 0.7815175522702191))) * S)) * ((0.07073588220193987 + (oo / t)) ** (P * P)))",
            "((((((((-0.4143934593308956 + (Z - (P - -1.0359989119430404))) - np.exp(P)) + A) * (h ** (P + ee))) - 0.4411245021335934) * (P ** 1.34247611756834)) * -0.0011400308784582946) + 0.04779878468835499)",
            "((np.exp((-0.5287541136581156 / 1.542599901612986) - ((((0.8868912212695856 + -0.01038483690643215) * K) - 3.134393705374266) ** A)) - ((-0.01038483690643215 * np.exp(S)) * np.exp(K / 1.214843936962893))) - 0.9377251169374243)",
        ]
        self.max_index = len(self.exprs_list)
        self.expr_dict = {i: self.exprs_list[i] for i in range(self.max_index)}

        self.optimal_params_mean = np.array(
            [
                0.99981366,
                0.98741602,
                0.93160026,
                1.25239084,
                0.89176424,
                1.14216409,
                1.18282589,
                1.03099573,
                0.97350482,
                1.51548953,
                1.21417095,
                1.22029288,
                1.44589542,
            ]
        )
        self.optimal_params_std = np.array(
            [
                7.33480329e-05,
                7.11806534e-03,
                3.69362896e-02,
                8.81680867e-02,
                1.02847980e-01,
                1.01208746e-01,
                1.02696271e-01,
                1.20665450e-01,
                2.09444776e-01,
                1.40806118e-01,
                1.83535877e-01,
                2.81730164e-01,
                2.15989585e-01,
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
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]
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
        ax = np.sqrt(x)
        h = Z / N
        K = A ** (1 / 3)
        S = (N - Z) / A
        t = A ** (2 / 3)

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

    def predict_term(self, Z, N, term):
        feats = self._get_features(Z, N)
        return self.predict_symb_terms(feats, term)

    def predict_single_term(self, Z, N, index):
        term = self.get_expresion_term(index)
        return self.predict_term(Z, N, term)

    def predict_with_uncertainty_boostrapping(
        self,
        Z,
        N,
        index,
        N_ITERATIONS=1_000,
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
        extra_term = self.predict_term(Z, N, self.get_expresion_term(index))
        unc = np.mean(np.abs(preds - preds.mean(axis=0)), axis=0) + np.abs(extra_term)
        return preds.mean(), unc


sr_be = SrBe()
