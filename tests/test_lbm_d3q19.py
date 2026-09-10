"""
Unit-Tests für den 3D D3Q19 LBM Solver.
"""

import pytest
import numpy as np
from wake_sim.config import SimulationConfig
from wake_sim.solver.lbm_d3q19 import LBMD3Q19Solver
from wake_sim.solver.numba_kernels import (
    D3Q19_CX, D3Q19_CY, D3Q19_CZ, D3Q19_WEIGHTS
)


def test_d3q19_lattice_constants():
    """Testet die 19 diskreten Richtungen und Gewichte des D3Q19-Gitters."""
    assert len(D3Q19_CX) == 19
    assert len(D3Q19_CY) == 19
    assert len(D3Q19_CZ) == 19
    assert len(D3Q19_WEIGHTS) == 19
    assert np.isclose(np.sum(D3Q19_WEIGHTS), 1.0, atol=1e-12)


def test_d3q19_solver_execution():
    """Testet die Durchführung von 3D-Zeitschritten und Querschnitts-Extraktion."""
    config = SimulationConfig(
        nx=30,
        ny=24,
        nz=20,
        domain_length_m=15.0,
        domain_height_m=10.0,
        domain_width_m=12.0,
        total_steps=10
    )
    solver = LBMD3Q19Solver(
        config,
        wingspan_m=6.0,
        chord_length_m=1.5,
        inflow_speed_m_s=50.0
    )

    assert solver.current_step == 0
    solver.step(5)
    assert solver.current_step == 5
    assert solver.time_phys > 0.0

    # Querschnitt der Wirbelstärke extrahieren
    omega_x = solver.get_slice_vorticity(slice_axis="x", index=-2)
    assert omega_x.shape == (solver.ny, solver.nz)
    assert not np.isnan(np.sum(omega_x))

    omega_z = solver.get_slice_vorticity(slice_axis="z", index=solver.nz // 2)
    assert omega_z.shape == (solver.ny, solver.nx)
    assert not np.isnan(np.sum(omega_z))
