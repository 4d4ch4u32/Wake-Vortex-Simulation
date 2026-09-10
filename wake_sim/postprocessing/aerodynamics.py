"""
Berechnung aerodynamischer Kenngrößen: Zirkulationsprofil, Tangentialgeschwindigkeit, Enstrophie.
"""

from typing import Tuple, List
import numpy as np


def compute_circulation_profile(
    vorticity: np.ndarray,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    center: Tuple[float, float],
    radii: np.ndarray,
    dx: float,
    dy: float
) -> np.ndarray:
    r"""
    Berechnet das radiale Zirkulationsprofil Gamma(r) durch Flächenintegration der Wirbelstärke:
    Gamma(r) = \iint_{|x - x_0| <= r} omega(x, y) dA.

    :param vorticity: 2D-Array der Wirbelstärke [1/s]
    :param x_mesh: 2D-Gitter der X-Koordinaten [m]
    :param y_mesh: 2D-Gitter der Y-Koordinaten [m]
    :param center: Wirbelzentrum (x0, y0) [m]
    :param radii: 1D-Array der Auswerteradien [m]
    :param dx: Gitterabstand in X [m]
    :param dy: Gitterabstand in Y [m]
    :return: 1D-Array der Zirkulation Gamma(r) [m^2/s]
    """
    r_dist2 = (x_mesh - center[0]) ** 2 + (y_mesh - center[1]) ** 2
    gamma_profile = np.zeros_like(radii, dtype=np.float64)
    cell_area = dx * dy

    for i, r in enumerate(radii):
        mask = r_dist2 <= (r ** 2)
        gamma_profile[i] = np.sum(vorticity[mask]) * cell_area

    return gamma_profile


def compute_tangential_velocity_profile(
    ux: np.ndarray,
    uy: np.ndarray,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    center: Tuple[float, float],
    radii: np.ndarray,
    dr: float = None
) -> np.ndarray:
    """
    Berechnet die azimutal gemittelte Tangentialgeschwindigkeit v_theta(r).
    v_theta = -ux * sin(theta) + uy * cos(theta).
    """
    dx_rel = x_mesh - center[0]
    dy_rel = y_mesh - center[1]
    r_dist = np.sqrt(dx_rel ** 2 + dy_rel ** 2)

    with np.errstate(divide='ignore', invalid='ignore'):
        sin_theta = np.where(r_dist > 1e-9, dy_rel / r_dist, 0.0)
        cos_theta = np.where(r_dist > 1e-9, dx_rel / r_dist, 1.0)

    v_theta_2d = -ux * sin_theta + uy * cos_theta
    v_theta_avg = np.zeros_like(radii, dtype=np.float64)

    if dr is None and len(radii) > 1:
        dr = (radii[1] - radii[0]) / 2.0
    elif dr is None:
        dr = 1.0

    for i, r in enumerate(radii):
        ring_mask = (r_dist >= r - dr) & (r_dist < r + dr)
        if np.any(ring_mask):
            v_theta_avg[i] = np.mean(v_theta_2d[ring_mask])

    return v_theta_avg


def compute_enstrophy(vorticity: np.ndarray, dx: float, dy: float) -> float:
    r"""
    Berechnet die Gesamt-Enstrophie: E = 0.5 * \iint omega^2 dA [m^2 / s^2].
    """
    return 0.5 * np.sum(vorticity ** 2) * dx * dy
