"""
Unit-Tests für die Postprocessing- und Analyse-Module.
"""

import pytest
import numpy as np
from wake_sim.config import AIRCRAFT_PRESETS
from wake_sim.postprocessing.aerodynamics import (
    compute_circulation_profile,
    compute_tangential_velocity_profile,
    compute_enstrophy
)
from wake_sim.postprocessing.vortex_tracker import VortexTracker, VortexTrajectory
from wake_sim.postprocessing.metrics import (
    get_icao_category,
    compute_hazard_distance,
    compute_energy_decay
)


def test_enstrophy_and_circulation():
    """Testet die Berechnung von Enstrophie und radialem Zirkulationsprofil."""
    nx, ny = 50, 50
    x_1d = np.linspace(-10, 10, nx)
    y_1d = np.linspace(-10, 10, ny)
    xx, yy = np.meshgrid(x_1d, y_1d)

    # Gaußscher Wirbel
    gamma_test = 100.0
    rc = 2.0
    r2 = xx ** 2 + yy ** 2
    vort = (gamma_test / (np.pi * rc ** 2)) * np.exp(-r2 / (rc ** 2))

    dx = x_1d[1] - x_1d[0]
    dy = y_1d[1] - y_1d[0]

    # Gesamt-Zirkulation sollte nahe an gamma_test liegen
    radii = np.array([1.0, 3.0, 8.0])
    circ_prof = compute_circulation_profile(vort, xx, yy, center=(0, 0), radii=radii, dx=dx, dy=dy)
    assert circ_prof[-1] > 0.9 * gamma_test

    enstrophy = compute_enstrophy(vort, dx, dy)
    assert enstrophy > 0.0


def test_vortex_trajectory_and_rebound():
    """Testet Trajektorienberechnung und Bodeneffekt-Rebound-Erkennung."""
    times = np.linspace(0, 10, 11)
    # Simuliere Absinken und Wiederaufsteigen (Rebound)
    # y = (t - 5)^2 + 10 -> Minimum bei t=5 mit y=10, Anstieg danach bis y=35
    left_y = (times - 5.0) ** 2 + 10.0
    right_y = (times - 5.0) ** 2 + 10.0
    left_x = -15.0 - 0.5 * times
    right_x = 15.0 + 0.5 * times
    circ = np.full(11, 200.0)
    vort_max = np.full(11, 15.0)

    traj = VortexTrajectory(
        time=times,
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        circulation=circ,
        max_vorticity=vort_max
    )

    assert len(traj.separation) == 11
    assert np.all(traj.separation > 0.0)
    
    rebound = traj.detect_ground_rebound()
    assert rebound["rebound_detected"] is True
    assert np.isclose(rebound["min_height_m"], 10.0)
    assert np.isclose(rebound["time_at_min_height_s"], 5.0)


def test_icao_hazard_metrics():
    """Testet ICAO-Kategorisierung und Staffelungsabstände."""
    heavy_ac = AIRCRAFT_PRESETS["B777"]
    medium_ac = AIRCRAFT_PRESETS["A320"]
    light_ac = AIRCRAFT_PRESETS["C172"]

    assert get_icao_category(heavy_ac) == "Heavy"
    assert get_icao_category(medium_ac) == "Medium"
    assert get_icao_category(light_ac) == "Light"

    haz_heavy = compute_hazard_distance(heavy_ac, follower_category="Light")
    assert haz_heavy["separation_nm"] == 6.0
    assert haz_heavy["separation_m"] == 6.0 * 1852.0

    haz_medium = compute_hazard_distance(medium_ac, follower_category="Medium")
    assert haz_medium["separation_nm"] == 3.0
