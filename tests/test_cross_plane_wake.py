"""
Unit- und Integrationstests für den 2D Cross-Plane Wirbelschleppensolver.
"""

import pytest
import numpy as np
from wake_sim.config import SimulationConfig, AIRCRAFT_PRESETS
from wake_sim.solver.cross_plane_wake import CrossPlaneWakeSolver


@pytest.mark.parametrize("preset_name", ["A320", "B737", "B777", "C172"])
def test_cross_plane_solver_presets(preset_name):
    """Testet die fehlerfreie Initialisierung und Ausführung für alle Flugzeugmuster."""
    ac = AIRCRAFT_PRESETS[preset_name]
    config = SimulationConfig(
        aircraft_name=preset_name,
        nx=100,
        ny=70,
        domain_width_m=ac.wingspan * 3.0,
        domain_height_m=ac.wingspan * 2.0,
        initial_altitude=ac.wingspan * 1.5,
        total_steps=30
    )
    solver = CrossPlaneWakeSolver(config, aircraft=ac)

    assert solver.current_step == 0
    solver.step(15)
    assert solver.current_step == 15

    pos_l, pos_r = solver.track_vortex_cores()
    # Linker Wirbel sollte links von der Mitte liegen, rechter Wirbel rechts
    assert pos_l[0] < 0.0
    assert pos_r[0] > 0.0

    # Wirbelhöhe sollte positiv (über Grund) sein
    assert pos_l[1] > 0.0
    assert pos_r[1] > 0.0


def test_crosswind_effect():
    """Testet den Einfluss von Seitenwind auf die Wirbeldrift."""
    ac = AIRCRAFT_PRESETS["A320"]
    
    # Ohne Wind
    cfg_nowind = SimulationConfig(aircraft_name="A320", crosswind=0.0, nx=100, ny=70, total_steps=20)
    s_nowind = CrossPlaneWakeSolver(cfg_nowind, aircraft=ac)
    s_nowind.step(20)
    pos_l0, pos_r0 = s_nowind.track_vortex_cores()
    
    # Mit starkem Seitenwind nach rechts (+4 m/s)
    cfg_wind = SimulationConfig(aircraft_name="A320", crosswind=4.0, nx=100, ny=70, total_steps=20)
    s_wind = CrossPlaneWakeSolver(cfg_wind, aircraft=ac)
    s_wind.step(20)
    pos_l_wind, pos_r_wind = s_wind.track_vortex_cores()

    # Mit Seitenwind nach rechts sollten die Wirbel weiter rechts liegen
    mean_x_nowind = 0.5 * (pos_l0[0] + pos_r0[0])
    mean_x_wind = 0.5 * (pos_l_wind[0] + pos_r_wind[0])
    assert mean_x_wind > mean_x_nowind
