"""
CFD-Solver-Paket: D2Q9, D3Q19 und Cross-Plane Wirbelschleppen-Solver.
"""

from .lbm_d2q9 import LBMD2Q9Solver
from .cross_plane_wake import CrossPlaneWakeSolver
from .lbm_d3q19 import LBMD3Q19Solver

__all__ = ["LBMD2Q9Solver", "CrossPlaneWakeSolver", "LBMD3Q19Solver"]
