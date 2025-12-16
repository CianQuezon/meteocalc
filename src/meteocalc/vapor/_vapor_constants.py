"""
Constants for different Saturation Vapor Equations.

Implements:
- Goff & Gratch

Author: Cian Quezon
"""
import numpy as np

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class GoffGratchConstants:
    T_ref: np.float64
    P_ref: np.float64
    A: np.float64
    B: np.float64
    C: np.float64
    C_exp: np.float64
    D: np.float64
    D_exp: np.float64
    log_p_ref = np.float64


