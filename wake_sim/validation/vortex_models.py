"""
Analytische Wirbelmodelle: Lamb-Oseen, Burnham-Hallock und Spiegelladungsmethode im Bodeneffekt.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


@dataclass
class LambOseenVortex:
    """
    Exakte viskose Lösung der Navier-Stokes-Gleichungen für einen isolierten Wirbel.
    v_theta(r, t) = (Gamma_0 / (2 * pi * r)) * (1 - exp(-r^2 / (4 * nu * t + rc^2)))
    """
    gamma_0: float                   # Initiale Gesamtzirkulation [m^2/s]
    core_radius_0: float             # Initialer Kernradius r_c0 [m]
    kinematic_viscosity: float = 1.5e-5 # Kinematische Viskosität nu [m^2/s]

    def effective_core_radius(self, time: float) -> float:
        """Effektiver Kernradius r_c(t) = sqrt(4 * nu * t + r_c0^2)."""
        return float(np.sqrt(4.0 * self.kinematic_viscosity * time + self.core_radius_0 ** 2))

    def tangential_velocity(self, r: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Berechnet das Tangentialgeschwindigkeitsprofil v_theta(r, t) [m/s]."""
        rc_t2 = 4.0 * self.kinematic_viscosity * time + self.core_radius_0 ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            term = 1.0 - np.exp(- (r ** 2) / rc_t2)
            v_theta = np.where(r > 1e-12, (self.gamma_0 / (2.0 * np.pi * r)) * term, 0.0)
        return v_theta

    def vorticity(self, r: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Berechnet die Wirbelstärkeverteilung omega_z(r, t) [1/s]."""
        rc_t2 = 4.0 * self.kinematic_viscosity * time + self.core_radius_0 ** 2
        return (self.gamma_0 / (np.pi * rc_t2)) * np.exp(- (r ** 2) / rc_t2)

    def circulation_profile(self, r: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Berechnet die Zirkulation innerhalb des Radius r: Gamma(r) [m^2/s]."""
        rc_t2 = 4.0 * self.kinematic_viscosity * time + self.core_radius_0 ** 2
        return self.gamma_0 * (1.0 - np.exp(- (r ** 2) / rc_t2))


@dataclass
class BurnhamHallockVortex:
    """
    Standard-Industriemodell (FAA / NASA) für Flugzeug-Wirbelschleppen.
    v_theta(r) = (Gamma_0 / (2 * pi)) * (r / (r^2 + rc^2))
    """
    gamma_0: float                   # Gesamtzirkulation [m^2/s]
    core_radius: float               # Kernradius rc [m]

    def tangential_velocity(self, r: np.ndarray) -> np.ndarray:
        """Tangentialgeschwindigkeit v_theta(r) [m/s]."""
        return (self.gamma_0 / (2.0 * np.pi)) * (r / (r ** 2 + self.core_radius ** 2))

    def vorticity(self, r: np.ndarray) -> np.ndarray:
        """Wirbelstärke omega_z(r) [1/s]."""
        return (self.gamma_0 / np.pi) * (self.core_radius ** 2) / ((r ** 2 + self.core_radius ** 2) ** 2)

    def circulation_profile(self, r: np.ndarray) -> np.ndarray:
        """Zirkulation Gamma(r) = Gamma_0 * r^2 / (r^2 + rc^2) [m^2/s]."""
        return self.gamma_0 * (r ** 2) / (r ** 2 + self.core_radius ** 2)


@dataclass
class GroundEffectImageModel:
    """
    Analytische Trajektorie eines Wirbelpaars im Bodeneffekt (Spiegelladungsmethode).
    Reale Wirbel bei (+/- b0/2, h), Spiegelwirbel bei (+/- b0/2, -h).
    """
    gamma_0: float                   # Zirkulation Gamma_0 [m^2/s]
    initial_spacing: float           # Wirbelabstand b_0 [m]
    initial_altitude: float          # Anfangshöhe h_0 [m]

    def descent_velocity(self, height: float) -> float:
        """
        Vertikale Absinkgeschwindigkeit -dh/dt [m/s] als Funktion der Höhe h:
        w(h) = (Gamma_0 / (2 * pi * b0)) * (1 - 1 / (1 + 4 * (h / b0)^2)).
        """
        b0 = self.initial_spacing
        h_ratio = height / max(b0, 1e-6)
        w0 = self.gamma_0 / (2.0 * np.pi * b0)
        return float(w0 * (1.0 - 1.0 / (1.0 + 4.0 * (h_ratio ** 2))))

    def lateral_drift_velocity(self, height: float) -> float:
        """
        Horizontale Driftgeschwindigkeit dx/dt [m/s] eines Wirbels im Bodeneffekt:
        v_drift(h) = (Gamma_0 / (2 * pi * b0)) * (2 * (h / b0) / (1 + 4 * (h / b0)^2)).
        """
        b0 = self.initial_spacing
        h_ratio = height / max(b0, 1e-6)
        w0 = self.gamma_0 / (2.0 * np.pi * b0)
        return float(w0 * (2.0 * h_ratio / (1.0 + 4.0 * (h_ratio ** 2))))

    def compute_trajectory(
        self,
        t_max: float = 60.0,
        dt: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Integrierte analytische Trajektorie (Höhe h(t), Wirbelabstand b(t), Geschwindigkeiten).
        """
        n_steps = int(t_max / dt)
        time = np.linspace(0.0, t_max, n_steps)
        h = np.zeros(n_steps)
        x_right = np.zeros(n_steps)
        
        h[0] = self.initial_altitude
        x_right[0] = self.initial_spacing / 2.0
        
        for i in range(1, n_steps):
            w = self.descent_velocity(h[i - 1])
            v_drift = self.lateral_drift_velocity(h[i - 1])
            
            # Euler-Integration
            h[i] = max(h[i - 1] - w * dt, 0.5)  # Mindesthöhe
            x_right[i] = x_right[i - 1] + v_drift * dt

        x_left = -x_right
        
        return {
            "time": time,
            "height": h,
            "left_x": x_left,
            "right_x": x_right,
            "separation": 2.0 * x_right,
            "descent_velocity": np.array([self.descent_velocity(hi) for hi in h]),
            "lateral_velocity": np.array([self.lateral_drift_velocity(hi) for hi in h])
        }
