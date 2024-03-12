####
# SR guided for correction of the binding energy
from typing import Any, List
import numpy as np


class SRBEModels:
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
            # "np.exp(-0.8948709195716154 * (((((0.9270832840257225 ** (N + N)) * N) * ((0.9270832840257225 ** N) * N)) + (0.02994986947801994 * x)) + ((((P ** 1.9412112817066696) * (h ** t)) / 0.46865314891483706) - 1.2343874050146584)))",
            # "((-0.4788113460721299 + ((((0.012167385837240422 - ((-2.047639966100053 / h) - S)) + (K * K)) * ((P * P) ** (K * 0.7815175522702191))) * S)) * ((0.07073588220193987 + (oo / t)) ** (P * P)))",
            # "((((((((-0.4143934593308956 + (Z - (P - -1.0359989119430404))) - np.exp(P)) + A) * (h ** (P + ee))) - 0.4411245021335934) * (P ** 1.34247611756834)) * -0.0011400308784582946) + 0.04779878468835499)",
            # "((np.exp((-0.5287541136581156 / 1.542599901612986) - ((((0.8868912212695856 + -0.01038483690643215) * K) - 3.134393705374266) ** A)) - ((-0.01038483690643215 * np.exp(S)) * np.exp(K / 1.214843936962893))) - 0.9377251169374243)",
            # "((((((np.exp(P) * (((t / -0.23680052314399586) * np.exp(K)) * np.exp(2.4101110036495026))) * np.exp(0.3024939640586434)) / np.exp(A)) * ((np.exp(P) / -0.814964922771754) + ee)) * ((np.exp(K) * x) * x)) * x)",
            # "(-0.0001435325624834701 * ((((np.exp(N) - (np.exp(N) - x)) * ee) * N) - -0.14882302746736192))",
            # "np.exp((((d - (K + (0.20012509200934805 - (h - ((P - (0.6877218764407389 - 0.6855985808423883)) * x))))) + 0.5300799768762136) * 1.5778867052836374) + h)",
            # "((((np.exp(P) / h) * eo) * np.exp(-0.18988241370166287 * Z)) - np.exp(((-0.18018060012176534 + -0.09575946147664494) * Z) + (eo - -1.7413864226030702)))",
        ]

    @staticmethod
    def get_features(Z, N):
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
        return {
            "Z": Z,
            "A": A,
            "N": N,
            "x": x,
            "ax": ax,
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

    def predict_be(self, Z: int, N: int, expression: str):
        """Predict the binding energy of a nucleus using the given expression"""
        features = self.get_features(Z, N)
        return eval(expression, None, features)

    def predict_sp(self, Z: int, N: int, f: callable):
        pred_be_this = f(Z, N)
        pred_be_up = f(Z + 1, N)
        return pred_be_up - pred_be_this

    def predict_sn(self, Z: int, N: int, f: callable):
        pred_be_this = f(Z, N)
        pred_be_up = f(Z, N - 1)
        return pred_be_up - pred_be_this

    def get_expression(self, max_index):
        return " + ".join(self.exprs_list[:max_index])

    def get_model(self, max_index):
        st = self.get_expression(max_index).replace("x", "(N - Z) ** 2")
        st = st.replace("h", "Z / N")
        st = st.replace("K", "A ** (1 / 3)")
        st = st.replace("S", "(N - Z) / A")
        st = st.replace("t", "A ** (2 / 3)")
        return st

    def __call__(self, Z, N, index=-1, ensemble: List[callable] = []):
        assert index < len(
            self.exprs_list
        ), "index must be less than the length of the learned expansion"
        expression = self.get_expression(index)
        pred = self.predict_be(Z, N, expression)
        if ensemble:
            for model in ensemble:
                pred += model(Z, N) / len(ensemble)
        return pred


sr_be = SRBEModels()
