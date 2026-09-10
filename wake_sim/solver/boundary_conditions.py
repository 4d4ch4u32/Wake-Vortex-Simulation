"""
Randbedingungen für 2D- und 3D-LBM-Simulationen:
Zou-He Einströmung/Ausströmung, No-Slip Wand (Boden / Tragfläche) und periodische Ränder.
"""

import numpy as np
from .numba_kernels import (
    HAS_NUMBA, njit,
    D2Q9_CX, D2Q9_CY, D2Q9_WEIGHTS, D2Q9_OPPOSITE,
    d2q9_equilibrium_node
)


@njit(fastmath=True)
def apply_zou_he_inflow_left(f, ux_in, uy_in, ny):
    """
    Zou-He Geschwindigkeits-Randbedingung am linken Rand (x = 0).
    Vorgegebene Einströmgeschwindigkeit (ux_in, uy_in).
    """
    for y in range(ny):
        # f0, f2, f4 sind tangential, f3, f6, f7 kommen von innen
        f0 = f[0, y, 0]
        f2 = f[2, y, 0]
        f4 = f[4, y, 0]
        f3 = f[3, y, 0]
        f6 = f[6, y, 0]
        f7 = f[7, y, 0]

        rho_w = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_in)

        # Unbekannte Verteilungen (1, 5, 8), die nach rechts zeigen
        f[1, y, 0] = f3 + (2.0 / 3.0) * rho_w * ux_in
        f[5, y, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_w * ux_in + 0.5 * rho_w * uy_in
        f[8, y, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_w * ux_in - 0.5 * rho_w * uy_in


@njit(fastmath=True)
def apply_zou_he_outflow_right(f, rho_out, ny, nx):
    """
    Zou-He Druck-/Dichte-Randbedingung am rechten Rand (x = nx - 1).
    Fester Ausströmdruck p_out bzw. Dichte rho_out (z.B. 1.0).
    """
    for y in range(ny):
        f0 = f[0, y, nx - 1]
        f2 = f[2, y, nx - 1]
        f4 = f[4, y, nx - 1]
        f1 = f[1, y, nx - 1]
        f5 = f[5, y, nx - 1]
        f8 = f[8, y, nx - 1]

        ux_w = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
        uy_w = 0.0

        # Unbekannte Verteilungen (3, 6, 7), die nach links zeigen
        f[3, y, nx - 1] = f1 - (2.0 / 3.0) * rho_out * ux_w
        f[6, y, nx - 1] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux_w + 0.5 * rho_out * uy_w
        f[7, y, nx - 1] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux_w - 0.5 * rho_out * uy_w


@njit(fastmath=True)
def apply_convective_outflow(f, f_prev, ux_conv, ny, nx):
    """
    Konvektive Ausström-Randbedingung 1. Ordnung für offene Ränder:
    df/dt + U_conv * df/dx = 0 -> f^{t+1}(nx-1) = f^t(nx-1) - U_conv * (f^t(nx-1) - f^t(nx-2)).
    """
    for y in range(ny):
        for i in range(9):
            f[i, y, nx - 1] = f_prev[i, y, nx - 1] - ux_conv * (f_prev[i, y, nx - 1] - f_prev[i, y, nx - 2])


@njit(fastmath=True)
def apply_top_slip_boundary(f, ny, nx):
    """
    Reibungsfreie Slip-Randbedingung am oberen Rand (y = ny - 1):
    Spiegelung der vertikalen Geschwindigkeitsanteile.
    """
    for x in range(nx):
        # 4 (unten) spiegelt sich aus 2 (oben), etc.
        f[4, ny - 1, x] = f[2, ny - 1, x]
        f[7, ny - 1, x] = f[6, ny - 1, x]
        f[8, ny - 1, x] = f[5, ny - 1, x]


@njit(fastmath=True)
def apply_ground_noslip_boundary(f, nx):
    """
    Feste No-Slip Wand (Boden) bei y = 0 mittels Standard-Bounce-Back.
    """
    for x in range(nx):
        # 2 nach oben aus 4 (nach unten)
        # 5 nach oben-rechts aus 7 (nach unten-links)
        # 6 nach oben-links aus 8 (nach unten-rechts)
        f2 = f[4, 0, x]
        f5 = f[7, 0, x]
        f6 = f[8, 0, x]

        f[2, 0, x] = f2
        f[5, 0, x] = f5
        f[6, 0, x] = f6
