"""
3D D3Q19 LBM Solver für 3D-Flügelumströmung und Randwirbelbildung (Wingtip Vortices).
"""

from typing import Tuple, Dict, List, Optional
import numpy as np

from ..config import SimulationConfig, AircraftParameters
from ..preprocessing.airfoil import NACAProfile
from ..preprocessing.lattice_units import LatticeConverter
from ..preprocessing.grid import Grid3D
from .numba_kernels import (
    HAS_NUMBA, njit,
    D3Q19_CX, D3Q19_CY, D3Q19_CZ, D3Q19_WEIGHTS, D3Q19_OPPOSITE,
    d3q19_collide_stream_step, d3q19_update_macroscopic
)


class LBMD3Q19Solver:
    """
    3D Lattice Boltzmann Solver (D3Q19) zur Modellierung von Randwirbeln und 3D-Nachlauf.
    """

    def __init__(
        self,
        config: SimulationConfig,
        wingspan_m: float = 8.0,
        chord_length_m: float = 2.0,
        inflow_speed_m_s: float = 60.0
    ):
        self.config = config
        self.nx = config.nx // 2 if config.nx > 80 else config.nx
        self.ny = config.ny // 2 if config.ny > 60 else config.ny
        self.nz = config.nz
        
        self.lx = config.domain_length_m
        self.ly = config.domain_height_m
        self.lz = config.domain_width_m
        
        self.span_phys = wingspan_m
        self.chord_phys = chord_length_m
        self.u_inflow_phys = inflow_speed_m_s
        self.density_phys = config.density
        
        # Einheitenumrechner
        self.converter = LatticeConverter(
            length_phys=self.chord_phys,
            length_lb=max(float(self.nx) * (self.chord_phys / self.lx), 6.0),
            velocity_phys=self.u_inflow_phys,
            velocity_lb=config.max_lattice_velocity,
            density_phys=self.density_phys,
            kinematic_viscosity_phys=config.kinematic_viscosity,
            target_reynolds=min(config.reynolds_number, 500.0)
        )
        
        self.dx_phys = self.lx / max(1, self.nx - 1)
        self.dy_phys = self.ly / max(1, self.ny - 1)
        self.dz_phys = self.lz / max(1, self.nz - 1)
        self.dt_phys = self.converter.dt
        
        # 3D Gitterfelder
        self.f = np.zeros((19, self.ny, self.nx, self.nz), dtype=np.float64)
        self.f_post = np.zeros((19, self.ny, self.nx, self.nz), dtype=np.float64)
        self.rho = np.ones((self.ny, self.nx, self.nz), dtype=np.float64)
        self.ux = np.zeros((self.ny, self.nx, self.nz), dtype=np.float64)
        self.uy = np.zeros((self.ny, self.nx, self.nz), dtype=np.float64)
        self.uz = np.zeros((self.ny, self.nx, self.nz), dtype=np.float64)
        
        # Geometriegitter
        self.grid = Grid3D(
            nx=self.nx, ny=self.ny, nz=self.nz,
            lx=self.lx, ly=self.ly, lz=self.lz,
            ground_boundary=True
        )
        
        # Flügel einfügen (Mitte der Domäne)
        airfoil = NACAProfile(code="0012", chord=self.chord_phys, angle_of_attack_deg=6.0)
        self.solid_mask = self.grid.add_wing(
            airfoil=airfoil,
            span=self.span_phys,
            root_position=(0.2 * self.lx, 0.5 * self.ly, 0.5 * self.lz)
        )
        
        self.current_step = 0
        self.time_phys = 0.0
        self.u_inflow_lb = self.converter.velocity_lb
        
        self._initialize_flow()

    def _initialize_flow(self):
        """Initialisiert das 3D-Strömungsfeld."""
        self.ux.fill(self.u_inflow_lb)
        self.uy.fill(0.0)
        self.uz.fill(0.0)
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0
        self.uz[self.solid_mask] = 0.0
        self.rho.fill(1.0)
        
        u_sq = self.ux ** 2 + self.uy ** 2 + self.uz ** 2
        for i in range(19):
            cu = D3Q19_CX[i] * self.ux + D3Q19_CY[i] * self.uy + D3Q19_CZ[i] * self.uz
            self.f[i] = D3Q19_WEIGHTS[i] * self.rho * (
                1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq
            )
        self.f_post = np.copy(self.f)

    def step(self, n_steps: int = 1):
        """Führt n_steps 3D LBM-Zeitschritte aus."""
        omega = self.converter.relaxation_frequency
        
        for _ in range(n_steps):
            d3q19_collide_stream_step(
                self.f, self.f_post, self.rho,
                self.ux, self.uy, self.uz,
                self.solid_mask, omega,
                self.nx, self.ny, self.nz
            )
            
            # Einströmung links (x=0)
            u_in = self.u_inflow_lb
            for i in range(19):
                cu = D3Q19_CX[i] * u_in
                feq = D3Q19_WEIGHTS[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * (u_in ** 2))
                self.f[i, :, 0, :] = feq
                
            d3q19_update_macroscopic(
                self.f, self.rho,
                self.ux, self.uy, self.uz,
                self.solid_mask,
                self.nx, self.ny, self.nz
            )
            
            self.current_step += 1
            self.time_phys += self.dt_phys

    def get_slice_vorticity(self, slice_axis: str = "x", index: int = -1) -> np.ndarray:
        """
        Extrahiert einen 2D-Querschnitt der Wirbelstärke entlang einer Achse.
        Standard: Nachlauf-Querschnitt quer zur Flugrichtung (Y-Z-Ebene bei x_idx).
        """
        if slice_axis == "x":
            # Y-Z Querschnitt bei gegebenem x
            idx = self.nx - 2 if index == -1 else index
            # Wirbelstärke omega_x = duz/dy - duy/dz
            uy_slice = self.converter.vel_to_phys(self.uy[:, idx, :])
            uz_slice = self.converter.vel_to_phys(self.uz[:, idx, :])
            
            duz_dy = np.gradient(uz_slice, self.dy_phys, axis=0)
            duy_dz = np.gradient(uy_slice, self.dz_phys, axis=1)
            omega_x = duz_dy - duy_dz
            return omega_x
        elif slice_axis == "z":
            # X-Y Flügelschnitt in Spannweitenmitte
            idx = self.nz // 2 if index == -1 else index
            ux_slice = self.converter.vel_to_phys(self.ux[:, :, idx])
            uy_slice = self.converter.vel_to_phys(self.uy[:, :, idx])
            
            duy_dx = np.gradient(uy_slice, self.dx_phys, axis=1)
            dux_dy = np.gradient(ux_slice, self.dy_phys, axis=0)
            omega_z = duy_dx - dux_dy
            return omega_z
        else:
            raise ValueError(f"Unbekannte Achse {slice_axis}")
