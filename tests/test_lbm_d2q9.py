"""
Unit-Tests für den 2D D2Q9 LBM Solver.
"""

import pytest
import numpy as np
from wake_sim.config import SimulationConfig
from wake_sim.solver.lbm_d2q9 import LBMD2Q9Solver
from wake_sim.solver.numba_kernels import (
    d2q9_equilibrium_node,
    D2Q9_CX, D2Q9_CY, D2Q9_WEIGHTS
)


def test_d2q9_equilibrium_moments():
    """Testet die Erhaltung von Dichte und Impuls in der Gleichgewichtsverteilung."""
    rho = 1.05
    ux = 0.08
    uy = -0.04
    feq = d2q9_equilibrium_node(rho, ux, uy)

    # 0. Moment: Dichte
    assert np.isclose(np.sum(feq), rho, atol=1e-12)

    # 1. Moment: Impuls
    assert np.isclose(np.sum(feq * D2Q9_CX), rho * ux, atol=1e-12)
    assert np.isclose(np.sum(feq * D2Q9_CY), rho * uy, atol=1e-12)


def test_d2q9_solver_execution():
    """Testet die Durchführung von Simulationsschritten und Kraftberechnung."""
    config = SimulationConfig(
        nx=80,
        ny=50,
        domain_width_m=20.0,
        domain_height_m=10.0,
        total_steps=50
    )
    solver = LBMD2Q9Solver(
        config,
        airfoil_code="0012",
        chord_length_m=2.0,
        angle_of_attack_deg=5.0,
        inflow_speed_m_s=60.0
    )

    assert solver.current_step == 0
    solver.step(20)
    assert solver.current_step == 20
    assert solver.time_phys > 0.0

    # Aerodynamische Kräfte
    l, d, cl, cd = solver.compute_aerodynamic_forces()
    assert isinstance(cl, float)
    assert isinstance(cd, float)
    assert not np.isnan(cl)
    assert not np.isnan(cd)

    # Wirbelstärkefeld
    vort = solver.get_vorticity_phys()
    assert vort.shape == (50, 80)
    assert not np.isnan(np.sum(vort))
