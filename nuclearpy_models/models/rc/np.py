###
# Implementation of the model described in: https://arxiv.org/abs/nucl-th/9401015
import numpy as np


class NPModel:
    """Nerlo-Pomorska, B.; Pomorski, K. A simple formula for nuclear charge radius"""

    def __init__(self, params=None):
        if not params:
            self.params = {
                "ro": 1.2347,
                "a": 0.1428,
                "b": 1.4378,
            }

    @staticmethod
    def _compute(Z, N, A, ro, a, b):
        fact = 1 - a * (N - Z) / A + b / A
        return np.sqrt(3 / 5) * ro * (A ** (1 / 3)) * fact

    def __call__(self, Z, N):
        A = Z + N
        return self._compute(Z, N, A, **self.params)


np_rc = NPModel()
