"""
Unit-Tests für den NACA-Profilgenerator und die Gitter-Rasterisierung.
"""

import pytest
import numpy as np
from wake_sim.preprocessing.airfoil import NACAProfile, generate_naca_4digit


def test_naca0012_geometry():
    """Testet die Symmetrie und Geometrie von NACA 0012."""
    foil = NACAProfile(code="0012", chord=1.0, n_points=100)
    assert foil.m == 0.0
    assert foil.p == 0.0
    assert foil.t == 0.12
    assert len(foil.x_coords) > 0
    assert len(foil.y_coords) > 0

    # Bei symmetrischem Profil ohne Anstellwinkel sollte der Schwerpunkt bei y=0 liegen
    assert np.isclose(np.mean(foil.y_coords), 0.0, atol=1e-3)


def test_naca2412_camber():
    """Testet gewölbtes NACA 2412 Profil."""
    foil = NACAProfile(code="2412", chord=2.0, n_points=150)
    assert foil.m == 0.02
    assert foil.p == 0.4
    assert foil.t == 0.12
    assert foil.chord == 2.0
    # Bei positivem Wölbungswert sollte das Profil im Mittel nach oben gewölbt sein
    assert np.max(foil.y_coords) > np.abs(np.min(foil.y_coords))


def test_angle_of_attack_rotation():
    """Testet die Drehung des Profils mit Anstellwinkel."""
    foil_0 = NACAProfile(code="0012", chord=1.0, angle_of_attack_deg=0.0)
    foil_10 = NACAProfile(code="0012", chord=1.0, angle_of_attack_deg=10.0)

    # Bei 10° Anstellwinkel (Nase nach oben) muss die Nase (x=0) höhere Y-Werte haben
    # und die Hinterkante (x=1) tiefere Y-Werte
    assert foil_10.y_coords[0] < foil_0.y_coords[0] or foil_10.y_coords[-1] < foil_0.y_coords[-1]


def test_grid_rasterization():
    """Testet die Voxel-/Gittermaskenerzeugung."""
    foil = NACAProfile(code="0012", chord=0.5, center=(0.5, 0.5))
    x = np.linspace(0, 1.0, 100)
    y = np.linspace(0, 1.0, 100)
    xx, yy = np.meshgrid(x, y)

    mask = foil.rasterize_on_grid(xx, yy)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    assert np.any(mask)  # Mindestens einige Punkte müssen innerhalb des Profils liegen
    assert not np.all(mask)  # Nicht alle Punkte


def test_invalid_naca_code():
    """Testet Fehlerbehandlung bei ungültigen NACA-Codes."""
    with pytest.raises(ValueError):
        NACAProfile(code="12")
    with pytest.raises(ValueError):
        NACAProfile(code="ABCD")
