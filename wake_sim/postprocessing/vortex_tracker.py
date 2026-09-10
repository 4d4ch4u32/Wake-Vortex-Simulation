"""
Wirbelkerndetektion, Trajektorien-Tracking und Bodeneffekt-Analyse.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class VortexTrajectory:
    """
    Speichert die zeitliche Trajektorie eines Wirbelpaars (Backbord / Steuerbord).
    """
    time: np.ndarray
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    circulation: np.ndarray
    max_vorticity: np.ndarray

    @property
    def separation(self) -> np.ndarray:
        """Wirbelabstand b(t) = x_right(t) - x_left(t) [m]."""
        return self.right_x - self.left_x

    @property
    def mean_altitude(self) -> np.ndarray:
        """Mittlere Wirbelhöhe h(t) = 0.5 * (y_left + y_right) [m]."""
        return 0.5 * (self.left_y + self.right_y)

    @property
    def descent_rate(self) -> np.ndarray:
        """Absinkgeschwindigkeit w(t) = -dh/dt [m/s]."""
        if len(self.time) < 2:
            return np.zeros_like(self.time)
        return -np.gradient(self.mean_altitude, self.time)

    @property
    def lateral_drift_rate(self) -> np.ndarray:
        """Auseinanderdrift-Geschwindigkeit im Bodeneffekt v_drift(t) = 0.5 * d(b)/dt [m/s]."""
        if len(self.time) < 2:
            return np.zeros_like(self.time)
        return 0.5 * np.gradient(self.separation, self.time)

    def detect_ground_rebound(self) -> Dict[str, float]:
        """
        Analysiert, ob und wann ein Bodeneffekt-Rebound (Wiederaufsteigen der Wirbel) auftritt.
        """
        h = self.mean_altitude
        if len(h) < 5:
            return {"rebound_detected": False, "min_height_m": float(h[0]) if len(h) else 0.0}

        min_idx = int(np.argmin(h))
        min_h = float(h[min_idx])
        t_min = float(self.time[min_idx])

        # Falls nach dem Minimum die Höhe wieder ansteigt
        if min_idx < len(h) - 1:
            max_after = float(np.max(h[min_idx:]))
            rebound_height = max_after - min_h
            rebound_detected = rebound_height > 0.5  # mehr als 0.5 m Wiederanstieg
        else:
            rebound_height = 0.0
            rebound_detected = False

        return {
            "rebound_detected": rebound_detected,
            "min_height_m": min_h,
            "time_at_min_height_s": t_min,
            "rebound_gain_m": rebound_height
        }


class VortexTracker:
    """
    Klasse zur sukzessiven Verfolgung der Wirbelkerne während einer Simulation.
    """

    def __init__(self):
        self.times: List[float] = []
        self.lx_list: List[float] = []
        self.ly_list: List[float] = []
        self.rx_list: List[float] = []
        self.ry_list: List[float] = []
        self.circ_list: List[float] = []
        self.vort_max_list: List[float] = []

    def update(
        self,
        time: float,
        pos_left: Tuple[float, float],
        pos_right: Tuple[float, float],
        circulation: float,
        max_vorticity: float
    ):
        """Fügt einen neuen Datenpunkt zur Trajektorie hinzu."""
        self.times.append(time)
        self.lx_list.append(pos_left[0])
        self.ly_list.append(pos_left[1])
        self.rx_list.append(pos_right[0])
        self.ry_list.append(pos_right[1])
        self.circ_list.append(circulation)
        self.vort_max_list.append(max_vorticity)

    def get_trajectory(self) -> VortexTrajectory:
        """Gibt das aggregierte VortexTrajectory-Objekt zurück."""
        return VortexTrajectory(
            time=np.array(self.times),
            left_x=np.array(self.lx_list),
            left_y=np.array(self.ly_list),
            right_x=np.array(self.rx_list),
            right_y=np.array(self.ry_list),
            circulation=np.array(self.circ_list),
            max_vorticity=np.array(self.vort_max_list)
        )
