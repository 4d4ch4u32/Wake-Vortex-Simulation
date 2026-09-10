"""
Numerische Hochleistungs-Rechenkerne für D2Q9- und D3Q19-LBM-Solver.
Unterstützt Numba-JIT-Beschleunigung mit automatischem Fallback auf vektorisiertes NumPy.
"""

import numpy as np

# Prüfe Verfügbarkeit von Numba
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy Dekoratoren
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ============================================================================
# D2Q9 Konstanten
# ============================================================================
D2Q9_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
D2Q9_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
D2Q9_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
D2Q9_OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


# ============================================================================
# D2Q9 Numba JIT Kerne
# ============================================================================

@njit(fastmath=True)
def d2q9_equilibrium_node(rho_val, ux_val, uy_val):
    """Berechnet die 9 Gleichgewichts-Verteilungsfunktionen für einen einzelnen Gitterpunkt."""
    feq = np.empty(9, dtype=np.float64)
    u_sq = ux_val * ux_val + uy_val * uy_val

    for i in range(9):
        cu = D2Q9_CX[i] * ux_val + D2Q9_CY[i] * uy_val
        feq[i] = D2Q9_WEIGHTS[i] * rho_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
    return feq


@njit(fastmath=True)
def d2q9_collide_and_stream(f, f_post, rho, ux, uy, solid_mask, omega, nx, ny):
    """
    Kombinierter BGK-Kollisions-, Bounce-Back- und Streaming-Schritt für D2Q9.
    """
    # 1. Kollision & Bounce-Back in temporäres Array f_post
    for y in range(ny):
        for x in range(nx):
            if solid_mask[y, x]:
                # Standard No-Slip Bounce-Back an Festkörpern
                for i in range(9):
                    f_post[D2Q9_OPPOSITE[i], y, x] = f[i, y, x]
            else:
                rho_val = rho[y, x]
                ux_val = ux[y, x]
                uy_val = uy[y, x]
                u_sq = ux_val * ux_val + uy_val * uy_val

                for i in range(9):
                    cu = D2Q9_CX[i] * ux_val + D2Q9_CY[i] * uy_val
                    feq = D2Q9_WEIGHTS[i] * rho_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
                    f_post[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq)

    # 2. Streaming von f_post zurück in f
    for i in range(9):
        cx = D2Q9_CX[i]
        cy = D2Q9_CY[i]
        for y in range(ny):
            next_y = (y + cy) % ny
            for x in range(nx):
                next_x = (x + cx) % nx
                f[i, next_y, next_x] = f_post[i, y, x]


@njit(fastmath=True)
def d2q9_update_macroscopic(f, rho, ux, uy, solid_mask, nx, ny):
    """Aktualisiert Dichte und Geschwindigkeitsfeld aus den Verteilungsfunktionen."""
    for y in range(ny):
        for x in range(nx):
            if solid_mask[y, x]:
                rho[y, x] = 1.0
                ux[y, x] = 0.0
                uy[y, x] = 0.0
            else:
                rho_sum = 0.0
                ux_sum = 0.0
                uy_sum = 0.0
                for i in range(9):
                    fi = f[i, y, x]
                    rho_sum += fi
                    ux_sum += fi * D2Q9_CX[i]
                    uy_sum += fi * D2Q9_CY[i]

                rho[y, x] = rho_sum
                if rho_sum > 1e-12:
                    ux[y, x] = ux_sum / rho_sum
                    uy[y, x] = uy_sum / rho_sum
                else:
                    ux[y, x] = 0.0
                    uy[y, x] = 0.0


@njit(fastmath=True)
def compute_vorticity_2d(ux, uy, dx, dy, nx, ny):
    """
    Berechnet die Wirbelstärke omega_z = duy/dx - dux/dy mit zentralen Differenzen 2. Ordnung.
    """
    vort = np.zeros((ny, nx), dtype=np.float64)

    # Innere Gitterpunkte (zentrale Differenzen)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            duy_dx = (uy[y, x + 1] - uy[y, x - 1]) / (2.0 * dx)
            dux_dy = (ux[y + 1, x] - ux[y - 1, x]) / (2.0 * dy)
            vort[y, x] = duy_dx - dux_dy

    # Ränder: Einseitige Differenzen
    for y in range(ny):
        vort[y, 0] = (uy[y, 1] - uy[y, 0]) / dx - (ux[min(y + 1, ny - 1), 0] - ux[max(y - 1, 0), 0]) / (2.0 * dy)
        vort[y, nx - 1] = (uy[y, nx - 1] - uy[y, nx - 2]) / dx - (ux[min(y + 1, ny - 1), nx - 1] - ux[max(y - 1, 0), nx - 1]) / (2.0 * dy)

    for x in range(nx):
        vort[0, x] = (uy[0, min(x + 1, nx - 1)] - uy[0, max(x - 1, 0)]) / (2.0 * dx) - (ux[1, x] - ux[0, x]) / dy
        vort[ny - 1, x] = (uy[ny - 1, min(x + 1, nx - 1)] - uy[ny - 1, max(x - 1, 0)]) / (2.0 * dx) - (ux[ny - 1, x] - ux[ny - 2, x]) / dy

    return vort


# ============================================================================
# D3Q19 Konstanten & Kerne
# ============================================================================
D3Q19_CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int32)
D3Q19_CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int32)
D3Q19_CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int32)

D3Q19_WEIGHTS = np.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36
], dtype=np.float64)

D3Q19_OPPOSITE = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], dtype=np.int32)


@njit(fastmath=True)
def d3q19_collide_stream_step(f, f_post, rho, ux, uy, uz, solid_mask, omega, nx, ny, nz):
    """
    3D D3Q19 LBM Kollision, Bounce-Back und Streaming.
    """
    for y in range(ny):
        for x in range(nx):
            for z in range(nz):
                if solid_mask[y, x, z]:
                    for i in range(19):
                        f_post[D3Q19_OPPOSITE[i], y, x, z] = f[i, y, x, z]
                else:
                    r = rho[y, x, z]
                    u = ux[y, x, z]
                    v = uy[y, x, z]
                    w = uz[y, x, z]
                    u_sq = u * u + v * v + w * w

                    for i in range(19):
                        cu = D3Q19_CX[i] * u + D3Q19_CY[i] * v + D3Q19_CZ[i] * w
                        feq = D3Q19_WEIGHTS[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
                        f_post[i, y, x, z] = f[i, y, x, z] - omega * (f[i, y, x, z] - feq)

    # Streaming
    for i in range(19):
        cx = D3Q19_CX[i]
        cy = D3Q19_CY[i]
        cz = D3Q19_CZ[i]
        for y in range(ny):
            next_y = (y + cy) % ny
            for x in range(nx):
                next_x = (x + cx) % nx
                for z in range(nz):
                    next_z = (z + cz) % nz
                    f[i, next_y, next_x, next_z] = f_post[i, y, x, z]


@njit(fastmath=True)
def d3q19_update_macroscopic(f, rho, ux, uy, uz, solid_mask, nx, ny, nz):
    """Aktualisiert Dichte und 3D-Geschwindigkeiten."""
    for y in range(ny):
        for x in range(nx):
            for z in range(nz):
                if solid_mask[y, x, z]:
                    rho[y, x, z] = 1.0
                    ux[y, x, z] = 0.0
                    uy[y, x, z] = 0.0
                    uz[y, x, z] = 0.0
                else:
                    r_sum = 0.0
                    u_sum = 0.0
                    v_sum = 0.0
                    w_sum = 0.0
                    for i in range(19):
                        fi = f[i, y, x, z]
                        r_sum += fi
                        u_sum += fi * D3Q19_CX[i]
                        v_sum += fi * D3Q19_CY[i]
                        w_sum += fi * D3Q19_CZ[i]

                    rho[y, x, z] = r_sum
                    if r_sum > 1e-12:
                        ux[y, x, z] = u_sum / r_sum
                        uy[y, x, z] = v_sum / r_sum
                        uz[y, x, z] = w_sum / r_sum
                    else:
                        ux[y, x, z] = 0.0
                        uy[y, x, z] = 0.0
                        uz[y, x, z] = 0.0
