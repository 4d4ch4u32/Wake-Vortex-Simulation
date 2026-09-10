"""
Prandtl'sche Traglinientheorie (Lifting-Line Theory) zur analytischen Berechnung
von Auftriebsverteilung, Zirkulation und induziertem Abwind.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class PrandtlLiftingLine:
    """
    Analytisches Modell nach Ludwig Prandtl für endliche Tragflügel.
    """
    wingspan: float                  # Spannweite b [m]
    wing_area: float                 # Flügelfläche S [m^2]
    velocity: float                  # Fluggeschwindigkeit U_inf [m/s]
    air_density: float = 1.225       # Luftdichte rho [kg/m^3]
    lift_curve_slope_2d: float = 2.0 * np.pi # Profilauftriebsanstieg a0 = dcl/dalpha [1/rad] (2*pi für dünne Profile)
    zero_lift_angle_deg: float = 0.0 # Nullanstellwinkel alpha_0 [Grad]

    @property
    def aspect_ratio(self) -> float:
        """Flügelstreckung Lambda = b^2 / S [-]."""
        return (self.wingspan ** 2) / self.wing_area

    @property
    def mean_chord(self) -> float:
        """Mittlere Profilsehne c_mean = S / b [m]."""
        return self.wing_area / self.wingspan

    def lift_coefficient(self, angle_of_attack_deg: float, oswald_factor: float = 0.95) -> float:
        """
        Berechnet den 3D-Gesamtauftriebsbeiwert C_L nach Prandtl:
        C_L = (a0 * (alpha - alpha_0)) / (1 + a0 / (pi * Lambda * e)).
        """
        alpha_rad = np.radians(angle_of_attack_deg - self.zero_lift_angle_deg)
        denom = 1.0 + self.lift_curve_slope_2d / (np.pi * self.aspect_ratio * oswald_factor)
        return float((self.lift_curve_slope_2d * alpha_rad) / denom)

    def induced_drag_coefficient(self, cl: float, oswald_factor: float = 0.95) -> float:
        """
        Berechnet den induzierten Widerstandsbeiwert C_Di:
        C_Di = C_L^2 / (pi * Lambda * e).
        """
        return float((cl ** 2) / (np.pi * self.aspect_ratio * oswald_factor))

    def root_circulation(self, cl: float) -> float:
        """
        Berechnet die maximale Zirkulation in Flügelmitte (elliptische Auftriebsverteilung):
        Gamma_0 = (2 * C_L * U_inf * S) / (pi * b) = (2 * L) / (pi * rho * U_inf * b) [m^2/s].
        """
        return float((2.0 * cl * self.velocity * self.wing_area) / (np.pi * self.wingspan))

    def downwash_velocity(self, cl: float) -> float:
        """
        Berechnet den induzierten Abwind w_i am Flügel:
        w_i = (C_L / (pi * Lambda)) * U_inf [m/s].
        """
        return float((cl / (np.pi * self.aspect_ratio)) * self.velocity)

    def spanwise_circulation_distribution(
        self,
        cl: float,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gibt die elliptische Zirkulationsverteilung entlang der Spannweite y in [-b/2, b/2] zurück:
        Gamma(y) = Gamma_0 * sqrt(1 - (2y/b)^2).
        """
        y = np.linspace(-self.wingspan / 2.0, self.wingspan / 2.0, num_points)
        gamma_0 = self.root_circulation(cl)
        eta = (2.0 * y) / self.wingspan
        gamma = gamma_0 * np.sqrt(np.maximum(1.0 - eta ** 2, 0.0))
        return y, gamma

    def solve_general_wing(
        self,
        angle_of_attack_deg: float,
        n_modes: int = 15,
        taper_ratio: float = 1.0
    ) -> Dict[str, any]:
        """
        Löst die allgemeine Prandtl'sche Traglinien-Fourierreihe für trapezförmige Flügel:
        sum_n A_n * sin(n*theta) * [1 + (n*pi*b) / (a0 * c(theta))] = alpha(theta).
        """
        theta = np.linspace(np.pi / (2 * n_modes), np.pi - np.pi / (2 * n_modes), n_modes)
        y = - (self.wingspan / 2.0) * np.cos(theta)
        
        # Lokale Sehne c(y) bei Zuspitzung taper_ratio = c_tip / c_root
        c_root = (2.0 * self.wing_area) / (self.wingspan * (1.0 + taper_ratio))
        c_y = c_root * (1.0 - (1.0 - taper_ratio) * (2.0 * np.abs(y) / self.wingspan))
        
        alpha_rad = np.radians(angle_of_attack_deg - self.zero_lift_angle_deg)
        
        # Lineares Gleichungssystem M * A = RHS
        matrix = np.zeros((n_modes, n_modes), dtype=np.float64)
        rhs = np.full(n_modes, alpha_rad, dtype=np.float64)
        
        for i, th in enumerate(theta):
            for j in range(n_modes):
                n = 2 * j + 1  # Symmetrische Moden (n = 1, 3, 5, ...)
                matrix[i, j] = np.sin(n * th) * (1.0 + (n * np.pi * self.wingspan) / (self.lift_curve_slope_2d * c_y[i] * np.sin(th)))
                
        # Koeffizienten A_n
        A = np.linalg.solve(matrix, rhs)
        
        # Auftriebsbeiwert C_L = pi * Lambda * A_1
        cl = float(np.pi * self.aspect_ratio * A[0])
        
        # Induzierter Widerstandsfaktor delta = sum_{n>1} n * (A_n / A_1)^2
        delta = float(np.sum([((2 * j + 1) * (A[j] / A[0]) ** 2) for j in range(1, n_modes)]))
        cdi = float((cl ** 2) / (np.pi * self.aspect_ratio) * (1.0 + delta))
        
        # Zirkulationsverteilung
        gamma = 2.0 * self.wingspan * self.velocity * np.sum(
            [A[j] * np.sin((2 * j + 1) * theta) for j in range(n_modes)], axis=0
        )
        
        return {
            "y": y,
            "chord": c_y,
            "circulation": gamma,
            "cl": cl,
            "cdi": cdi,
            "oswald_factor": 1.0 / (1.0 + delta),
            "gamma_max": float(np.max(gamma))
        }
