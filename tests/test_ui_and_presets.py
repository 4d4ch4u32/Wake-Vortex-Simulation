"""
Unit-Tests für CLI, Flugzeug-Presets und Export-Routinen.
"""

import os
import pytest
import numpy as np
from wake_sim.config import AIRCRAFT_PRESETS, AircraftParameters, SimulationConfig
from wake_sim.ui.cli import parse_arguments
from wake_sim.solver.cross_plane_wake import CrossPlaneWakeSolver
from wake_sim.visualization.export import export_simulation_results


def test_aircraft_presets_physical_consistency():
    """Überprüft die physikalische Konsistenz aller Flugzeugmuster."""
    for name, ac in AIRCRAFT_PRESETS.items():
        assert ac.mass > 0.0
        assert ac.wingspan > 0.0
        assert ac.wing_area > 0.0
        assert ac.approach_speed > 0.0
        assert ac.weight > 0.0
        assert ac.aspect_ratio > 3.0
        assert ac.vortex_spacing > 0.5 * ac.wingspan
        assert ac.initial_circulation() > 0.0
        assert ac.initial_descent_rate() > 0.0
        assert ac.initial_core_radius() > 0.0


def test_cli_parsing():
    """Testet das Parsen von CLI-Argumenten."""
    args = parse_arguments(["--mode", "cross_plane", "--aircraft", "B777", "--height", "75.0", "--steps", "250"])
    assert args.mode == "cross_plane"
    assert args.aircraft == "B777"
    assert args.height == 75.0
    assert args.steps == 250


def test_csv_export(tmp_path):
    """Testet den Export von Telemetriedaten in eine CSV-Datei."""
    csv_file = str(tmp_path / "test_export.csv")
    config = SimulationConfig(nx=60, ny=40, total_steps=10)
    solver = CrossPlaneWakeSolver(config)
    for _ in range(10):
        solver.step(1)

    export_simulation_results(solver, csv_file)
    assert os.path.exists(csv_file)
    assert os.path.getsize(csv_file) > 100

    # Dateiinhalt prüfen
    with open(csv_file, "r") as f:
        header = f.readline()
        assert "time_s" in header
        assert "circulation_m2_s" in header
        lines = f.readlines()
        assert len(lines) >= 10
