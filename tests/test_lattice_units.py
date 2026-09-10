"""
Unit-Tests für die LBM-Einheitenumrechnung.
"""

import pytest
import numpy as np
from wake_sim.preprocessing.lattice_units import LatticeConverter


def test_lattice_converter_consistency():
    """Testet Hin- und Rücktransformation von Größen."""
    conv = LatticeConverter(
        length_phys=35.8,
        length_lb=100.0,
        velocity_phys=70.0,
        velocity_lb=0.08,
        density_phys=1.225,
        kinematic_viscosity_phys=1.5e-5
    )

    # Position
    x_phys = 15.0
    x_lb = conv.pos_to_lb(x_phys)
    assert np.isclose(conv.pos_to_phys(x_lb), x_phys)

    # Geschwindigkeit
    u_phys = 55.0
    u_lb = conv.vel_to_lb(u_phys)
    assert np.isclose(conv.vel_to_phys(u_lb), u_phys)

    # Zeit
    t_phys = 2.5
    t_lb = conv.time_to_lb(t_phys)
    assert np.isclose(conv.time_to_phys(t_lb), t_phys)

    # Zirkulation
    gamma_phys = 250.0
    gamma_lb = conv.circulation_to_lb(gamma_phys)
    assert np.isclose(conv.circulation_to_phys(gamma_lb), gamma_phys)


def test_stability_limits():
    """Testet, dass die Relaxationszeit tau immer > 0.5 ist."""
    conv = LatticeConverter(
        length_phys=10.0,
        length_lb=50.0,
        velocity_phys=50.0,
        velocity_lb=0.05,
        target_reynolds=1e6
    )
    assert conv.tau > 0.5
    assert conv.relaxation_frequency > 0.0
