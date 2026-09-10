"""
Export von Simulationsergebnissen als CSV-Dateien und animierte Grafiken (GIF/PNG).
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..solver.cross_plane_wake import CrossPlaneWakeSolver
from .live_plotter import LiveWakePlotter


def export_simulation_results(solver: CrossPlaneWakeSolver, filepath_csv: str):
    """
    Exportiert die zeitliche Historie der Simulation als CSV-Datei.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath_csv)), exist_ok=True)
    
    h = solver.history
    data = np.column_stack([
        h["time"],
        h["left_vortex_x"],
        h["left_vortex_y"],
        h["right_vortex_x"],
        h["right_vortex_y"],
        h["max_vorticity"],
        h["circulation"],
        h["kinetic_energy"]
    ])
    
    header = "time_s,left_x_m,left_y_m,right_x_m,right_y_m,max_vorticity_1_s,circulation_m2_s,kinetic_energy_j"
    np.savetxt(filepath_csv, data, delimiter=",", header=header, comments="", fmt="%.6e")


def export_animation_gif(
    solver: CrossPlaneWakeSolver,
    filepath_gif: str,
    n_frames: int = 40,
    steps_per_frame: int = 10,
    fps: int = 15
):
    """
    Erzeugt eine animierte GIF-Datei des Simulationsverlaufs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath_gif)), exist_ok=True)
    
    plotter = LiveWakePlotter(solver, interactive=False)
    
    def animate_step(frame):
        solver.step(steps_per_frame)
        plotter.update()
        return [plotter.im_vort, plotter.im_vel, plotter.line_traj_h, plotter.line_traj_sep]
        
    anim = FuncAnimation(
        plotter.fig, animate_step, frames=n_frames, interval=1000 // fps, blit=False
    )
    anim.save(filepath_gif, writer="pillow", fps=fps)
    plt.close(plotter.fig)
