"""
Gittererzeugung und Randmaskierungs-Klassen für 2D- und 3D-Simulationen.
"""

from typing import Tuple, Optional
import numpy as np
from .airfoil import NACAProfile


class Grid2D:
    """
    Strukturiertes 2D-Rechengitter für LBM-Simulationen.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float = 1.0,
        ly: float = 1.0,
        ground_boundary: bool = True
    ):
        """
        :param nx: Anzahl Gitterknoten in X-Richtung (Breite / Spannweiten-Achse / Strömungsrichtung)
        :param ny: Anzahl Gitterknoten in Y-Richtung (Höhe über Grund)
        :param lx: Physikalische Breite [m]
        :param ly: Physikalische Höhe [m]
        :param ground_boundary: Wenn True, ist die untere Zeile (y=0) ein no-slip Boden
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.ground_boundary = ground_boundary

        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)

        # 1D- und 2D-Koordinatenvektoren
        self.x_1d = np.linspace(0.0, self.lx, self.nx)
        self.y_1d = np.linspace(0.0, self.ly, self.ny)
        self.x, self.y = np.meshgrid(self.x_1d, self.y_1d)

        # Masken
        self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
        if self.ground_boundary:
            self.solid_mask[0, :] = True  # Fester Boden bei y = 0

    def add_airfoil(
        self,
        airfoil: NACAProfile,
        position: Tuple[float, float] = (0.2, 0.5)
    ) -> np.ndarray:
        """
        Fügt ein NACA-Tragflächenprofil in das Gitter ein.

        :param airfoil: Instanz von NACAProfile
        :param position: (x, y) Position der Flügelvorderkante in physikalischen Koordinaten [m]
        :return: Aktualisierte solid_mask
        """
        # Profil neu positionieren
        airfoil.center = position
        airfoil.x_coords, airfoil.y_coords = airfoil._generate_coordinates()
        
        foil_mask = airfoil.rasterize_on_grid(self.x, self.y)
        self.solid_mask |= foil_mask
        return self.solid_mask

    def add_obstacle_circle(self, center: Tuple[float, float], radius: float) -> np.ndarray:
        """Fügt ein kreisförmiges Hindernis ein."""
        dist2 = (self.x - center[0]) ** 2 + (self.y - center[1]) ** 2
        circle_mask = dist2 <= (radius ** 2)
        self.solid_mask |= circle_mask
        return self.solid_mask

    def get_inflow_nodes(self) -> np.ndarray:
        """Knoten am linken Einströmbereich (x = 0)."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[:, 0] = True
        mask[self.solid_mask] = False
        return mask

    def get_outflow_nodes(self) -> np.ndarray:
        """Knoten am rechten Ausströmbereich (x = lx)."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[:, -1] = True
        mask[self.solid_mask] = False
        return mask

    def get_top_nodes(self) -> np.ndarray:
        """Knoten am oberen Rand (y = ly)."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[-1, :] = True
        mask[self.solid_mask] = False
        return mask

    def get_ground_nodes(self) -> np.ndarray:
        """Knoten am Boden (y = 0)."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[0, :] = True
        return mask


class Grid3D:
    """
    Strukturiertes 3D-Rechengitter für LBM-D3Q19-Simulationen.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        lx: float = 1.0,
        ly: float = 1.0,
        lz: float = 1.0,
        ground_boundary: bool = True
    ):
        """
        :param nx: Anzahl Knoten in X-Richtung (Strömungsrichtung / Nachlauf)
        :param ny: Anzahl Knoten in Y-Richtung (Höhe)
        :param nz: Anzahl Knoten in Z-Richtung (Spannweiten-Richtung)
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.lx = float(lx)
        self.ly = float(ly)
        self.lz = float(lz)
        self.ground_boundary = ground_boundary

        self.dx = self.lx / max(1, self.nx - 1)
        self.dy = self.ly / max(1, self.ny - 1)
        self.dz = self.lz / max(1, self.nz - 1)

        self.solid_mask = np.zeros((self.ny, self.nx, self.nz), dtype=bool)
        if self.ground_boundary:
            self.solid_mask[0, :, :] = True  # Fester Boden bei y = 0

    def add_wing(
        self,
        airfoil: NACAProfile,
        span: float,
        root_position: Tuple[float, float, float] = (0.2, 0.5, 0.5)
    ) -> np.ndarray:
        """
        Fügt einen endlichen 3D-Flügel in das Gitter ein.
        """
        # Z-Bereich des Flügels
        z_min = root_position[2] - span / 2.0
        z_max = root_position[2] + span / 2.0

        # 2D-Schnitt des Profils in X-Y
        x_2d = np.linspace(0.0, self.lx, self.nx)
        y_2d = np.linspace(0.0, self.ly, self.ny)
        xx, yy = np.meshgrid(x_2d, y_2d)

        airfoil.center = (root_position[0], root_position[1])
        airfoil.x_coords, airfoil.y_coords = airfoil._generate_coordinates()
        foil_mask_2d = airfoil.rasterize_on_grid(xx, yy)

        z_1d = np.linspace(0.0, self.lz, self.nz)
        for k, z_val in enumerate(z_1d):
            if z_min <= z_val <= z_max:
                self.solid_mask[:, :, k] |= foil_mask_2d

        return self.solid_mask
