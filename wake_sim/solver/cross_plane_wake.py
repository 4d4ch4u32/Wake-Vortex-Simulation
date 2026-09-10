"""
2D Cross-Plane (Trefftz-Ebene) Navier-Stokes LBM Solver für Wirbelschleppen
beim Landeanflug unter Einfluss von Bodeneffekt und Seitenwind.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np

from ..config import AircraftParameters, SimulationConfig, AIR_DENSITY_DEFAULT
from ..preprocessing.lattice_units import LatticeConverter
from .numba_kernels import (
    HAS_NUMBA, njit,
    D2Q9_CX, D2Q9_CY, D2Q9_WEIGHTS, D2Q9_OPPOSITE,
    d2q9_collide_and_stream, d2q9_update_macroscopic, compute_vorticity_2d
)
from .boundary_conditions import apply_ground_noslip_boundary, apply_top_slip_boundary


class CrossPlaneWakeSolver:
    """
    Simuliert die zeitliche Entwicklung und das Absinken eines gegenläufigen Wirbelpaars
    im Querschnitt (y-z bzw. x-y quer zur Flugrichtung) während der Landung.
    Modelliert Bodeneffekt (Vortex Rebound, Sekundärwirbel) und Seitenwindeinfluss.
    """

    def __init__(
        self,
        config: SimulationConfig,
        aircraft: Optional[AircraftParameters] = None
    ):
        self.config = config
        self.aircraft = aircraft if aircraft is not None else config.get_aircraft()
        
        # Gitterdimensionen
        self.nx = config.nx
        self.ny = config.ny
        self.lx = config.domain_width_m
        self.ly = config.domain_height_m
        
        # Physikalische Parameter
        self.density_phys = config.density
        self.altitude_init = config.initial_altitude
        self.crosswind_phys = config.crosswind
        
        # Wirbelkenngrößen
        self.gamma_0_phys = self.aircraft.initial_circulation(self.density_phys)
        self.b0_phys = self.aircraft.vortex_spacing
        self.rc_phys = self.aircraft.initial_core_radius()
        self.w0_phys = self.aircraft.initial_descent_rate(self.density_phys)
        
        # Maximale Tangentialgeschwindigkeit im Wirbelkern: v_max = Gamma_0 / (4 * pi * rc)
        self.v_max_phys = self.gamma_0_phys / (4.0 * np.pi * max(self.rc_phys, 0.1))
        
        # Einheitenumrechnung (Referenzlänge = b_0, Referenzgeschwindigkeit = v_max)
        self.converter = LatticeConverter(
            length_phys=self.b0_phys,
            length_lb=float(self.nx) * (self.b0_phys / self.lx),
            velocity_phys=max(1.2 * self.v_max_phys, self.w0_phys, 5.0),
            velocity_lb=config.max_lattice_velocity,
            density_phys=self.density_phys,
            kinematic_viscosity_phys=config.kinematic_viscosity,
            target_reynolds=config.reynolds_number
        )
        
        self.dx_phys = self.lx / (self.nx - 1)
        self.dy_phys = self.ly / (self.ny - 1)
        self.dt_phys = self.converter.dt
        
        # Koordinatennetze in SI-Einheiten [m]
        # x zentriert um Landebahnmittellinie (-lx/2 bis +lx/2)
        self.x_phys_1d = np.linspace(-self.lx / 2.0, self.lx / 2.0, self.nx)
        self.y_phys_1d = np.linspace(0.0, self.ly, self.ny)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_phys_1d, self.y_phys_1d)
        
        # LBM Felder (Gittereinheiten)
        self.f = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        self.f_post = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        self.rho = np.ones((self.ny, self.nx), dtype=np.float64)
        self.ux = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.uy = np.zeros((self.ny, self.nx), dtype=np.float64)
        
        # Feste Wand am Boden (y = 0)
        self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.solid_mask[0, :] = True
        
        # Zeitschritt-Zähler & Historie
        self.current_step = 0
        self.time_phys = 0.0
        self.history: Dict[str, List[float]] = {
            "time": [],
            "left_vortex_x": [],
            "left_vortex_y": [],
            "right_vortex_x": [],
            "right_vortex_y": [],
            "max_vorticity": [],
            "circulation": [],
            "kinetic_energy": []
        }
        
        self._initialize_flow_field()

    def _burnham_hallock_velocity(
        self,
        x_target: np.ndarray,
        y_target: np.ndarray,
        x_core: float,
        y_core: float,
        gamma: float,
        rc: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet das Geschwindigkeitsfeld eines einzelnen Burnham-Hallock Wirbels.
        v_theta(r) = (Gamma / (2 * pi)) * (r / (r^2 + rc^2))
        """
        dx = x_target - x_core
        dy = y_target - y_core
        r2 = dx ** 2 + dy ** 2
        r = np.sqrt(r2)
        
        v_theta = (gamma / (2.0 * np.pi)) * (r / (r2 + rc ** 2))
        
        # Geschwindigkeitskomponenten: u = -v_theta * sin(theta), v = v_theta * cos(theta)
        # sin(theta) = dy / r, cos(theta) = dx / r
        with np.errstate(divide='ignore', invalid='ignore'):
            ux = np.where(r > 1e-9, -v_theta * (dy / r), 0.0)
            uy = np.where(r > 1e-9, v_theta * (dx / r), 0.0)
            
        return ux, uy

    def _initialize_flow_field(self):
        """
        Initialisiert das Wirbelpaar (Backbord/Steuerbord) mit Burnham-Hallock Profil
        plus optionalem Spiegelladungs-Term und Seitenwind.
        """
        # Wirbelpositionen: Links (-b0/2, h0), Rechts (+b0/2, h0)
        xl, yl = -self.b0_phys / 2.0, self.altitude_init
        xr, yr = self.b0_phys / 2.0, self.altitude_init
        
        # Linker Wirbel (Gegenuhrzeigersinn: +Gamma_0)
        ux_l, uy_l = self._burnham_hallock_velocity(
            self.x_mesh, self.y_mesh, xl, yl, +self.gamma_0_phys, self.rc_phys
        )
        
        # Rechter Wirbel (Uhrzeigersinn: -Gamma_0)
        ux_r, uy_r = self._burnham_hallock_velocity(
            self.x_mesh, self.y_mesh, xr, yr, -self.gamma_0_phys, self.rc_phys
        )
        
        # Superposition des physikalischen Geschwindigkeitsfeldes
        ux_total_phys = ux_l + ux_r + self.crosswind_phys
        uy_total_phys = uy_l + uy_r
        
        # Am festen Boden (y = 0) Geschwindigkeit auf 0 setzen
        ux_total_phys[0, :] = 0.0
        uy_total_phys[0, :] = 0.0
        
        # Umrechnung in LB-Gittereinheiten
        ux_lb = self.converter.vel_to_lb(ux_total_phys)
        uy_lb = self.converter.vel_to_lb(uy_total_phys)
        
        # Clamping für numerische LBM-Stabilität (u_LB <= 0.15)
        u_mag = np.sqrt(ux_lb ** 2 + uy_lb ** 2)
        max_limit = 0.14
        scale = np.where(u_mag > max_limit, max_limit / (u_mag + 1e-12), 1.0)
        self.ux = ux_lb * scale
        self.uy = uy_lb * scale
        self.rho.fill(1.0)
        
        # Initialisierung der Gleichgewichts-Verteilungen f_i = f_i^(eq)
        u_sq = self.ux ** 2 + self.uy ** 2
        for i in range(9):
            cu = D2Q9_CX[i] * self.ux + D2Q9_CY[i] * self.uy
            self.f[i] = D2Q9_WEIGHTS[i] * self.rho * (
                1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq
            )
            
        self.f_post = np.copy(self.f)
        self._record_telemetry()

    def step(self, n_steps: int = 1):
        """Führt n_steps LBM-Zeitschritte durch."""
        omega = self.converter.relaxation_frequency
        
        for _ in range(n_steps):
            # 1. BGK Kollision und Streaming
            d2q9_collide_and_stream(
                self.f, self.f_post, self.rho, self.ux, self.uy,
                self.solid_mask, omega, self.nx, self.ny
            )
            
            # 2. Randbedingungen
            apply_ground_noslip_boundary(self.f, self.nx)
            apply_top_slip_boundary(self.f, self.ny, self.nx)
            
            # 3. Makroskopische Variablen aktualisieren
            d2q9_update_macroscopic(
                self.f, self.rho, self.ux, self.uy,
                self.solid_mask, self.nx, self.ny
            )
            
            self.current_step += 1
            self.time_phys += self.dt_phys

        self._record_telemetry()

    def get_vorticity_phys(self) -> np.ndarray:
        """Berechnet das physikalische Wirbelstärkefeld omega_z [1/s]."""
        vort_lb = compute_vorticity_2d(
            self.ux, self.uy, 1.0, 1.0, self.nx, self.ny
        )
        return self.converter.vorticity_to_phys(vort_lb)

    def get_velocity_phys(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gibt die physikalischen Geschwindigkeitsfelder (ux, uy, Betrag u_mag) [m/s] zurück.
        """
        ux_phys = self.converter.vel_to_phys(self.ux)
        uy_phys = self.converter.vel_to_phys(self.uy)
        u_mag = np.sqrt(ux_phys ** 2 + uy_phys ** 2)
        return ux_phys, uy_phys, u_mag

    def get_pressure_phys(self) -> np.ndarray:
        """Gibt das physikalische Druckfeld [Pa] zurück."""
        return self.converter.pressure_to_phys(self.rho)

    def track_vortex_cores(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Detektiert die Positionen der beiden Wirbelkerne (links: max. positive Wirbelstärke,
        rechts: min. negative Wirbelstärke).
        """
        vort = self.get_vorticity_phys()
        
        # Linke Hälfte (x < 0) für linken Wirbel (+Omega)
        # Bodennahe Randeffekte (unterste 3 Zeilen) ausschließen
        mid_x = self.nx // 2
        y_cut = 3
        
        left_region = vort[y_cut:, :mid_x]
        right_region = vort[y_cut:, mid_x:]
        
        if left_region.size > 0 and right_region.size > 0:
            # Maximaler positiver Wirbel links
            idx_l = np.unravel_index(np.argmax(left_region), left_region.shape)
            yl_idx = idx_l[0] + y_cut
            xl_idx = idx_l[1]
            pos_left = (self.x_phys_1d[xl_idx], self.y_phys_1d[yl_idx])
            
            # Maximaler negativer Wirbel rechts
            idx_r = np.unravel_index(np.argmin(right_region), right_region.shape)
            yr_idx = idx_r[0] + y_cut
            xr_idx = idx_r[1] + mid_x
            pos_right = (self.x_phys_1d[xr_idx], self.y_phys_1d[yr_idx])
        else:
            pos_left = (-self.b0_phys / 2.0, self.altitude_init)
            pos_right = (self.b0_phys / 2.0, self.altitude_init)
            
        return pos_left, pos_right

    def _record_telemetry(self):
        """Zeichnet physikalische Diagnosewerte auf."""
        pos_l, pos_r = self.track_vortex_cores()
        vort = self.get_vorticity_phys()
        ux_p, uy_p, u_mag = self.get_velocity_phys()
        
        # Kinetische Gesamtenergie E_kin = 0.5 * rho * integral(u^2) dA
        e_kin = 0.5 * self.density_phys * np.sum(u_mag ** 2) * self.dx_phys * self.dy_phys
        
        # Zirkulation grob geschätzt über Integral der positiven Wirbelstärke links
        circ_left = np.sum(np.maximum(vort[:, :self.nx // 2], 0.0)) * self.dx_phys * self.dy_phys
        
        self.history["time"].append(self.time_phys)
        self.history["left_vortex_x"].append(pos_l[0])
        self.history["left_vortex_y"].append(pos_l[1])
        self.history["right_vortex_x"].append(pos_r[0])
        self.history["right_vortex_y"].append(pos_r[1])
        self.history["max_vorticity"].append(float(np.max(np.abs(vort))))
        self.history["circulation"].append(float(circ_left))
        self.history["kinetic_energy"].append(float(e_kin))
