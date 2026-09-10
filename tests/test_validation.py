"""
Unit-Tests für die analytischen Validierungsmodelle (Prandtl, Lamb-Oseen, Burnham-Hallock, Spiegelladungen).
"""

import pytest
import numpy as np
from wake_sim.config import AIRCRAFT_PRESETS
from wake_sim.validation.prandtl_lifting_line import PrandtlLiftingLine
from wake_sim.validation.vortex_models import LambOseenVortex, BurnhamHallockVortex, GroundEffectImageModel
from wake_sim.validation.benchmark import run_analytical_validation


def test_prandtl_lifting_line_formulas():
    """Testet die analytischen Gleichungen der Prandtl-Traglinientheorie."""
    prandtl = PrandtlLiftingLine(
        wingspan=35.8,
        wing_area=122.6,
        velocity=70.0,
        air_density=1.225
    )
    assert prandtl.aspect_ratio > 10.0

    # Auftriebsbeiwert bei 5 Grad
    cl = prandtl.lift_coefficient(5.0)
    assert 0.3 < cl < 0.8

    # Zirkulationsverteilung
    y, gamma = prandtl.spanwise_circulation_distribution(cl, num_points=50)
    assert len(y) == 50
    assert np.isclose(gamma[0], 0.0, atol=1e-3)  # An Flügelspitzen = 0
    assert np.isclose(gamma[-1], 0.0, atol=1e-3)
    assert gamma[25] > 0.0  # In Flügelmitte maximal


def test_vortex_models():
    """Testet Lamb-Oseen und Burnham-Hallock Wirbelprofile."""
    gamma0 = 250.0
    rc = 1.5
    radii = np.linspace(0.1, 20.0, 100)

    # Burnham-Hallock
    bh = BurnhamHallockVortex(gamma_0=gamma0, core_radius=rc)
    v_bh = bh.tangential_velocity(radii)
    # Maximalgeschwindigkeit sollte bei r = rc liegen
    r_max_idx = np.argmax(v_bh)
    assert np.isclose(radii[r_max_idx], rc, atol=0.3)
    assert np.isclose(np.max(v_bh), gamma0 / (4.0 * np.pi * rc), rtol=0.05)

    # Lamb-Oseen
    lo = LambOseenVortex(gamma_0=gamma0, core_radius_0=rc)
    v_lo = lo.tangential_velocity(radii, time=0.0)
    assert np.max(v_lo) > 0.0


def test_ground_effect_image_model():
    """Testet die Spiegelladungs-Trajektorienberechnung."""
    model = GroundEffectImageModel(gamma_0=250.0, initial_spacing=28.0, initial_altitude=50.0)
    traj = model.compute_trajectory(t_max=10.0, dt=0.1)

    assert len(traj["time"]) == 100
    # Wirbelhöhe sollte im freien Raum absinken
    assert traj["height"][-1] < traj["height"][0]
    # Abstand sollte sich im Bodeneffekt vergrößern
    assert traj["separation"][-1] >= traj["separation"][0]


def test_benchmark_runner():
    """Testet den Benchmark-Lauf und den Validierungsstatus."""
    val = run_analytical_validation("A320", sim_steps=40)
    assert val["status"] == "PASSED"
    assert val["profile_comparison"]["r2_burnham_hallock"] > 0.75
    assert val["circulation_relative_diff_percent"] < 1.0
