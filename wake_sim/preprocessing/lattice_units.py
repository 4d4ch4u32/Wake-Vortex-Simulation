"""
Einheitenumrechnung zwischen physikalischen SI-Einheiten und LBM-Gittereinheiten.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class LatticeConverter:
    """
    Konvertiert physikalische Größen (SI: m, s, kg, Pa, m^2/s) in
    dimensionslose Lattice-Boltzmann-Einheiten (LB: dx=1, dt=1, rho0=1) und umgekehrt.
    """
    length_phys: float               # Physikalische charakteristische Länge L_phys [m] (z.B. Spannweite b oder Sehne c)
    length_lb: float                 # Gitterpunkte entlang der charakteristischen Länge N_L [-]
    velocity_phys: float             # Physikalische Referenzgeschwindigkeit U_phys [m/s]
    velocity_lb: float = 0.08        # LB-Referenzgeschwindigkeit u_LB [-] (sollte <= 0.1 sein für Mach-Zahl < 0.17)
    density_phys: float = 1.225      # Physikalische Dichte rho_phys [kg/m^3]
    kinematic_viscosity_phys: float = 1.5e-5 # Physikalische Viskosität nu_phys [m^2/s]
    target_reynolds: float = None    # Optional: Ziel-Reynolds-Zahl zur numerischen Skalierung

    def __post_init__(self):
        # Gitterabstand dx [m]
        self.dx = self.length_phys / float(self.length_lb)
        # Zeitschritt dt [s] basierend auf u_LB = U_phys * (dt / dx)
        self.dt = self.velocity_lb * self.dx / self.velocity_phys
        
        # LB Schallgeschwindigkeit c_s = 1 / sqrt(3)
        self.cs = 1.0 / np.sqrt(3.0)
        self.cs2 = 1.0 / 3.0
        
        # Reynolds-Zahl
        if self.target_reynolds is not None and self.target_reynolds > 0:
            self.reynolds = float(self.target_reynolds)
            # nu_lb so berechnen, dass Re = u_LB * length_lb / nu_lb
            self.nu_lb = (self.velocity_lb * self.length_lb) / self.reynolds
        else:
            # Direkte Umrechnung der physikalischen Viskosität
            self.nu_lb = self.kinematic_viscosity_phys * self.dt / (self.dx ** 2)
            self.reynolds = (self.velocity_phys * self.length_phys) / self.kinematic_viscosity_phys

        # Relaxationszeit tau = 3 * nu_lb + 0.5
        self.tau = 3.0 * self.nu_lb + 0.5

        # Stabilitätsgrenzen prüfen
        if self.tau <= 0.5:
            # Automatische Stabilisierung für LBM BGK
            self.tau = 0.52
            self.nu_lb = (self.tau - 0.5) / 3.0
            self.reynolds = (self.velocity_lb * self.length_lb) / self.nu_lb

        # Mach-Zahl im Gitter Ma_LB = u_LB / c_s
        self.mach_lb = self.velocity_lb / self.cs

    @property
    def relaxation_frequency(self) -> float:
        """Kollisionsfrequenz omega = 1 / tau."""
        return 1.0 / self.tau

    # --- Umrechnungs-Methoden (LB -> SI) ---

    def pos_to_phys(self, x_lb: np.ndarray) -> np.ndarray:
        """Position / Länge [Gittereinheiten] -> [m]."""
        return x_lb * self.dx

    def vel_to_phys(self, u_lb: np.ndarray) -> np.ndarray:
        """Geschwindigkeit [LB] -> [m/s]."""
        return u_lb * (self.dx / self.dt)

    def time_to_phys(self, t_lb: float) -> float:
        """Zeit [LB Zeitschritte] -> [s]."""
        return t_lb * self.dt

    def vorticity_to_phys(self, omega_lb: np.ndarray) -> np.ndarray:
        """Wirbelstärke [1/dt] -> [1/s]."""
        return omega_lb / self.dt

    def pressure_to_phys(self, rho_lb: np.ndarray) -> np.ndarray:
        """Druck p = c_s^2 * (rho_lb - 1.0) -> Physikalischer dynamischer/statischer Druck [Pa]."""
        # p_phys = p_lb * rho_phys * (dx/dt)^2
        p_lb = self.cs2 * (rho_lb - 1.0)
        return p_lb * self.density_phys * ((self.dx / self.dt) ** 2)

    def circulation_to_phys(self, gamma_lb: float) -> float:
        """Zirkulation Gamma [LB: dx^2 / dt] -> [m^2/s]."""
        return gamma_lb * (self.dx ** 2) / self.dt

    # --- Umrechnungs-Methoden (SI -> LB) ---

    def pos_to_lb(self, x_phys: float) -> float:
        """Position / Länge [m] -> [Gittereinheiten]."""
        return x_phys / self.dx

    def vel_to_lb(self, u_phys: float) -> float:
        """Geschwindigkeit [m/s] -> [LB]."""
        return u_phys * (self.dt / self.dx)

    def time_to_lb(self, t_phys: float) -> float:
        """Zeit [s] -> [LB Zeitschritte]."""
        return t_phys / self.dt

    def circulation_to_lb(self, gamma_phys: float) -> float:
        """Zirkulation Gamma [m^2/s] -> [LB]."""
        return gamma_phys * self.dt / (self.dx ** 2)
