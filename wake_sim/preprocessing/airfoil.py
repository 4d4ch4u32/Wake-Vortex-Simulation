"""
NACA-4-Ziffern Profilgenerator und Gitter-Rasterisierung.
"""

from typing import Tuple, Optional
import numpy as np
from matplotlib.path import Path


class NACAProfile:
    """
    Generiert und transformiert NACA 4-Ziffern Tragflächenprofile (z. B. NACA 0012, NACA 2412).
    """

    def __init__(
        self,
        code: str = "0012",
        chord: float = 1.0,
        n_points: int = 200,
        angle_of_attack_deg: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0),
        closed_trailing_edge: bool = True
    ):
        """
        Initialisiert das NACA-Profil.

        :param code: 4-stelliger NACA-Code, z. B. '0012', '2412', '4412'
        :param chord: Sehnenlänge c [m]
        :param n_points: Anzahl der Stützpunkte entlang der Sehne
        :param angle_of_attack_deg: Anstellwinkel alpha [Grad] (positiv = Nase nach oben)
        :param center: Position der Profilsehnen-Vorderkante (x0, y0) [m]
        :param closed_trailing_edge: Geschlossene Hinterkante sicherstellen
        """
        self.code = code.upper().replace("NACA", "").strip()
        if len(self.code) != 4 or not self.code.isdigit():
            raise ValueError(f"Ungültiger NACA-Code '{code}'. Erwartet wird ein 4-stelliger Code wie '0012'.")

        self.chord = float(chord)
        self.n_points = n_points
        self.aoa_rad = np.radians(angle_of_attack_deg)
        self.center = center
        self.closed_trailing_edge = closed_trailing_edge

        # NACA Parameter parsen
        self.m = int(self.code[0]) / 100.0        # Maximale Wölbung
        self.p = int(self.code[1]) / 10.0         # Position der max. Wölbung
        self.t = int(self.code[2:4]) / 100.0      # Maximale relative Dicke

        self.x_coords, self.y_coords = self._generate_coordinates()

    def _generate_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet die Ober- und Unterseiten-Koordinaten nach analytischer NACA-Formel.
        """
        # Kosinus-Verteilung für höhere Punktedichte an Vorder- und Hinterkante
        beta = np.linspace(0.0, np.pi, self.n_points)
        x = self.chord * 0.5 * (1.0 - np.cos(beta))

        # Dickenverteilung y_t
        a4 = -0.1036 if self.closed_trailing_edge else -0.1015
        yt = 5.0 * self.t * self.chord * (
            0.2969 * np.sqrt(np.maximum(x / self.chord, 1e-12))
            - 0.1260 * (x / self.chord)
            - 0.3516 * (x / self.chord) ** 2
            + 0.2843 * (x / self.chord) ** 3
            + a4 * (x / self.chord) ** 4
        )

        # Skelettlinie (Camber line) und Ableitung
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        if self.m > 0 and self.p > 0:
            for i, xi in enumerate(x):
                xc = xi / self.chord
                if xc < self.p:
                    yc[i] = self.chord * (self.m / (self.p ** 2)) * (2.0 * self.p * xc - xc ** 2)
                    dyc_dx[i] = (2.0 * self.m / (self.p ** 2)) * (self.p - xc)
                else:
                    yc[i] = self.chord * (self.m / ((1.0 - self.p) ** 2)) * ((1.0 - 2.0 * self.p) + 2.0 * self.p * xc - xc ** 2)
                    dyc_dx[i] = (2.0 * self.m / ((1.0 - self.p) ** 2)) * (self.p - xc)

        theta = np.arctan(dyc_dx)

        # Ober- und Unterseite
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Zusammenhängendes Polygon: Von Hinterkante (oben) über Nase zu Hinterkante (unten)
        poly_x = np.concatenate([xu[::-1], xl[1:]])
        poly_y = np.concatenate([yu[::-1], yl[1:]])

        # Rotation um das vordere Viertelpunkt-Zentrum (oder Vorderkante) mit Anstellwinkel
        # Standard: Rotation um 1/4-Sehne (c/4)
        rot_cx = 0.25 * self.chord
        rot_cy = 0.0

        px_shifted = poly_x - rot_cx
        py_shifted = poly_y - rot_cy

        cos_a = np.cos(self.aoa_rad)
        sin_a = np.sin(self.aoa_rad)

        rot_x = cos_a * px_shifted + sin_a * py_shifted + rot_cx + self.center[0]
        rot_y = -sin_a * px_shifted + cos_a * py_shifted + rot_cy + self.center[1]

        return rot_x, rot_y

    def get_polygon(self) -> np.ndarray:
        """Gibt das 2D-Polygon als (N, 2)-Array zurück."""
        return np.column_stack((self.x_coords, self.y_coords))

    def rasterize_on_grid(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """
        Erzeugt eine binäre Festkörpermaske (True = Festkörper / Tragfläche) für 2D-Gitter.

        :param x_grid: 2D-Array der X-Koordinaten (Form: ny, nx)
        :param y_grid: 2D-Array der Y-Koordinaten (Form: ny, nx)
        :return: Boolesches 2D-Array (True = Hindernis)
        """
        polygon = self.get_polygon()
        path = Path(polygon)
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        mask_flat = path.contains_points(points)
        return mask_flat.reshape(x_grid.shape)


def generate_naca_4digit(
    code: str = "0012",
    chord: float = 1.0,
    angle_of_attack_deg: float = 0.0,
    center: Tuple[float, float] = (0.0, 0.0)
) -> NACAProfile:
    """Convenience-Funktion zur Erstellung eines NACA-Profils."""
    return NACAProfile(
        code=code,
        chord=chord,
        angle_of_attack_deg=angle_of_attack_deg,
        center=center
    )
