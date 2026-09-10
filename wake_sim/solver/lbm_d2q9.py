"""
2D D2Q9 LBM Solver für Tragflächenumströmung (NACA 0012) und Wirbelerzeugung.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np

from ..config import SimulationConfig, AircraftParameters
from ..preprocessing.airfoil import NACAProfile, generate_naca_4digit
from ..preprocessing.lattice_units import LatticeConverter
from .numba_kernels import (
    HAS_NUMBA, njit,
    D2Q9_CX, D2Q9_CY, D2Q9_WEIGHTS, D2Q9_OPPOSITE,
    d2q9_collide_and_stream, d2q9_update_macroscopic, compute_vorticity_2d
)
from .boundary_conditions import (
    apply_zou_he_inflow_left,
    apply_zou_he_outflow_right,
    apply_convective_outflow,
    apply_top_slip_boundary,
    apply_ground_noslip_boundary
)


class LBMD2Q9Solver:
    """
    Allgemeiner 2D D2Q9 LBM Solver für Tragflächenumströmungen (z.B. NACA 0012).
    Berechnet Strömungsfelder, Wirbelablösung und aerodynamische Beiwerte (CL, CD).
    """

    def __init__(
        self,
        config: SimulationConfig,
        airfoil_code: str = "0012",
        chord_length_m: float = 2.0,
        angle_of_attack_deg: float = 6.0,
        inflow_speed_m_s: float = 70.0
    ):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.lx = config.domain_width_m
        self.ly = config.domain_height_m
        
        self.chord_phys = chord_length_m
        self.aoa_deg = angle_of_attack_deg
        self.u_inflow_phys = inflow_speed_m_s
        self.density_phys = config.density
        
        # Einheitenumrechner
        # Flügelspannweite / Sehne als Referenzlänge
        chord_lb = float(self.nx) * (self.chord_phys / self.lx)
        self.converter = LatticeConverter(
            length_phys=self.chord_phys,
            length_lb=max(chord_lb, 10.0),
            velocity_phys=self.u_inflow_phys,
            velocity_lb=config.max_lattice_velocity,
            density_phys=self.density_phys,
            kinematic_viscosity_phys=config.kinematic_viscosity,
            target_reynolds=config.reynolds_number
        )
        
        self.dx_phys = self.lx / (self.nx - 1)
        self.dy_phys = self.ly / (self.ny - 1)
        self.dt_phys = self.converter.dt
        
        # Koordinaten
        self.x_phys_1d = np.linspace(0.0, self.lx, self.nx)
        self.y_phys_1d = np.linspace(0.0, self.ly, self.ny)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_phys_1d, self.y_phys_1d)
        
        # Gitterfelder
        self.f = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        self.f_post = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        self.f_prev = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        self.rho = np.ones((self.ny, self.nx), dtype=np.float64)
        self.ux = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.uy = np.zeros((self.ny, self.nx), dtype=np.float64)
        
        # Geometrie & Hindernismaske
        self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        # Tragflügel bei ca. 25% der Kanallänge und Kanalmitte platzieren
        foil_pos_x = 0.25 * self.lx
        foil_pos_y = 0.50 * self.ly
        self.airfoil = NACAProfile(
            code=airfoil_code,
            chord=self.chord_phys,
            n_points=200,
            angle_of_attack_deg=self.aoa_deg,
            center=(foil_pos_x, foil_pos_y)
        )
        self.foil_mask = self.airfoil.rasterize_on_grid(self.x_mesh, self.y_mesh)
        self.solid_mask |= self.foil_mask
        
        # Zeitschritt-Zähler & Historie
        self.current_step = 0
        self.time_phys = 0.0
        self.u_inflow_lb = self.converter.velocity_lb
        
        self.history: Dict[str, List[float]] = {
            "time": [],
            "cl": [],
            "cd": [],
            "max_vorticity": [],
            "kinetic_energy": []
        }
        
        self._initialize_flow()

    def _initialize_flow(self):
        """Initialisiert die Strömung mit gleichförmiger Einströmung."""
        self.ux.fill(self.u_inflow_lb)
        self.uy.fill(0.0)
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0
        self.rho.fill(1.0)
        
        u_sq = self.ux ** 2 + self.uy ** 2
        for i in range(9):
            cu = D2Q9_CX[i] * self.ux + D2Q9_CY[i] * self.uy
            self.f[i] = D2Q9_WEIGHTS[i] * self.rho * (
                1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq
            )
            
        self.f_post = np.copy(self.f)
        self.f_prev = np.copy(self.f)
        self._record_telemetry()

    def step(self, n_steps: int = 1):
        """Führt n_steps LBM-Zeitschritte durch."""
        omega = self.converter.relaxation_frequency
        
        for _ in range(n_steps):
            self.f_prev[:, :, :] = self.f[:, :, :]
            
            # 1. BGK Kollision und Streaming
            d2q9_collide_and_stream(
                self.f, self.f_post, self.rho, self.ux, self.uy,
                self.solid_mask, omega, self.nx, self.ny
            )
            
            # 2. Randbedingungen
            apply_zou_he_inflow_left(self.f, self.u_inflow_lb, 0.0, self.ny)
            apply_convective_outflow(self.f, self.f_prev, self.u_inflow_lb, self.ny, self.nx)
            apply_top_slip_boundary(self.f, self.ny, self.nx)
            
            # 3. Makroskopische Variablen
            d2q9_update_macroscopic(
                self.f, self.rho, self.ux, self.uy,
                self.solid_mask, self.nx, self.ny
            )
            
            self.current_step += 1
            self.time_phys += self.dt_phys

        self._record_telemetry()

    def get_vorticity_phys(self) -> np.ndarray:
        """Physikalisches Wirbelstärkefeld omega_z [1/s]."""
        vort_lb = compute_vorticity_2d(
            self.ux, self.uy, 1.0, 1.0, self.nx, self.ny
        )
        return self.converter.vorticity_to_phys(vort_lb)

    def get_velocity_phys(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Physikalische Geschwindigkeiten [m/s]."""
        ux_phys = self.converter.vel_to_phys(self.ux)
        uy_phys = self.converter.vel_to_phys(self.uy)
        u_mag = np.sqrt(ux_phys ** 2 + uy_phys ** 2)
        return ux_phys, uy_phys, u_mag

    def get_pressure_phys(self) -> np.ndarray:
        """Physikalisches Druckfeld [Pa]."""
        return self.converter.pressure_to_phys(self.rho)

    def compute_aerodynamic_forces(self) -> Tuple[float, float, float, float]:
        """
        Berechnet Auftriebskraft L [N/m], Widerstandskraft D [N/m] sowie
        die Beiwerte CL und CD mittels Impulsaustauschmethode (Momentum Exchange).
        """
        fx_lb = 0.0
        fy_lb = 0.0
        
        # Impulsaustausch an allen Festkörperknoten
        for y in range(1, self.ny - 1):
            for x in range(1, self.nx - 1):
                if self.foil_mask[y, x]:
                    for i in range(1, 9):
                        # Nachbarknoten in Richtung e_i
                        nx_node = x + D2Q9_CX[i]
                        ny_node = y + D2Q9_CY[i]
                        if not self.foil_mask[ny_node, nx_node]:
                            # Fluid-Knoten Nachbar
                            opp = D2Q9_OPPOSITE[i]
                            # Kraftbeitrag
                            f_val = self.f_post[i, y, x] + self.f[opp, ny_node, nx_node]
                            fx_lb += f_val * D2Q9_CX[i]
                            fy_lb += f_val * D2Q9_CY[i]

        # Umrechnung in physikalische Kräfte [N/m] (pro Meter Spannweite)
        # F_phys = F_lb * rho_phys * (dx^3 / dt^2)
        force_factor = self.density_phys * (self.converter.dx ** 3) / (self.converter.dt ** 2)
        drag_force = fx_lb * force_factor
        lift_force = fy_lb * force_factor
        
        # Dynamischer Druck q_inf = 0.5 * rho * U_inf^2
        q_inf = 0.5 * self.density_phys * (self.u_inflow_phys ** 2)
        ref_area = self.chord_phys * 1.0  # 1 m Spannweite
        
        cl = lift_force / max(q_inf * ref_area, 1e-6)
        cd = drag_force / max(q_inf * ref_area, 1e-6)
        
        return lift_force, drag_force, cl, cd

    def _record_telemetry(self):
        """Zeichnet aerodynamische Historie auf."""
        _, _, cl, cd = self.compute_aerodynamic_forces()
        vort = self.get_vorticity_phys()
        _, _, u_mag = self.get_velocity_phys()
        
        e_kin = 0.5 * self.density_phys * np.sum(u_mag ** 2) * self.dx_phys * self.dy_phys
        
        self.history["time"].append(self.time_phys)
        self.history["cl"].append(float(cl))
        self.history["cd"].append(float(cd))
        self.history["max_vorticity"].append(float(np.max(np.abs(vort))))
        self.history["kinetic_energy"].append(float(e_kin))
