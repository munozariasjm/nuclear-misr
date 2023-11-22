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
