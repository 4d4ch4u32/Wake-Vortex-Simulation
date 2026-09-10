"""
Matplotlib-Echtzeit-Plotter und statische Diagnose-Grafiken für Strömungsfelder.
"""

import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..solver.cross_plane_wake import CrossPlaneWakeSolver
from ..postprocessing.aerodynamics import compute_tangential_velocity_profile


class LiveWakePlotter:
    """
    Interaktiver Echtzeit-Plotter für die 2D/Cross-Plane Wirbelschleppen-Simulation.
    Zeigt 4 Dashboards: Wirbelstärke mit Stromlinien, Geschwindigkeitsbetrag,
    Wirbeltrajektorien und tangentiales Geschwindigkeitsprofil.
    """

    def __init__(
        self,
        solver: CrossPlaneWakeSolver,
        figsize: Tuple[int, int] = (14, 9),
        interactive: bool = True
    ):
        self.solver = solver
        self.interactive = interactive
        
        if self.interactive:
            plt.ion()

        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle(
            f"Wirbelschleppen-Simulation (CFD/LBM) - {solver.aircraft.name} (Landung)",
            fontsize=14, fontweight="bold"
        )

        self.ax_vort = self.axes[0, 0]
        self.ax_vel = self.axes[0, 1]
        self.ax_traj = self.axes[1, 0]
        self.ax_prof = self.axes[1, 1]

        # Initiale Grafik-Elemente
        self._init_plots()

    def _init_plots(self):
        """Initialisiert die Subplots und Colorbars."""
        s = self.solver
        x = s.x_mesh
        y = s.y_mesh
        vort = s.get_vorticity_phys()
        _, _, u_mag = s.get_velocity_phys()

        # 1. Wirbelstärke
        vmax_vort = max(float(np.max(np.abs(vort))), 1.0)
        self.im_vort = self.ax_vort.imshow(
            vort, extent=[s.x_phys_1d[0], s.x_phys_1d[-1], s.y_phys_1d[0], s.y_phys_1d[-1]],
            origin="lower", cmap="seismic", vmin=-vmax_vort, vmax=vmax_vort, aspect="auto"
        )
        self.ax_vort.axhline(0, color="k", linewidth=2.5, label="Landebahn (Boden)")
        self.ax_vort.set_title("Wirbelstärke $\\omega_z$ [$1/s$] & Wirbelkerne")
        self.ax_vort.set_xlabel("Spannweiten-Achse $x$ [m]")
        self.ax_vort.set_ylabel("Höhe über Grund $y$ [m]")
        self.cb_vort = self.fig.colorbar(self.im_vort, ax=self.ax_vort, orientation="vertical", pad=0.02)
        self.cb_vort.set_label("$\\omega_z$ [1/s]")

        # Wirbelkern-Marker
        pos_l, pos_r = s.track_vortex_cores()
        self.scatter_l, = self.ax_vort.plot([pos_l[0]], [pos_l[1]], "ko", markersize=8, label="Wirbelzentrum L")
        self.scatter_r, = self.ax_vort.plot([pos_r[0]], [pos_r[1]], "mo", markersize=8, label="Wirbelzentrum R")
        self.ax_vort.legend(loc="upper right", fontsize=8)

        # 2. Geschwindigkeit
        vmax_vel = max(float(np.max(u_mag)), 5.0)
        self.im_vel = self.ax_vel.imshow(
            u_mag, extent=[s.x_phys_1d[0], s.x_phys_1d[-1], s.y_phys_1d[0], s.y_phys_1d[-1]],
            origin="lower", cmap="plasma", vmin=0, vmax=vmax_vel, aspect="auto"
        )
        self.ax_vel.set_title("Geschwindigkeitsbetrag $|\\vec{u}|$ [m/s]")
        self.ax_vel.set_xlabel("Spannweiten-Achse $x$ [m]")
        self.ax_vel.set_ylabel("Höhe über Grund $y$ [m]")
        self.cb_vel = self.fig.colorbar(self.im_vel, ax=self.ax_vel, orientation="vertical", pad=0.02)
        self.cb_vel.set_label("$|\\vec{u}|$ [m/s]")

        # 3. Trajektorie (Höhe über Zeit)
        self.line_traj_h, = self.ax_traj.plot([], [], "b-", linewidth=2, label="Höhe $h(t)$")
        self.line_traj_sep, = self.ax_traj.plot([], [], "r--", linewidth=1.5, label="Wirbelabstand $b(t)$")
        self.ax_traj.set_title("Wirbelbahnen & Bodeneffekt")
        self.ax_traj.set_xlabel("Simulationszeit $t$ [s]")
        self.ax_traj.set_ylabel("Distanz [m]")
        self.ax_traj.grid(True, linestyle=":", alpha=0.6)
        self.ax_traj.legend(loc="upper right", fontsize=8)

        # 4. Tangentialgeschwindigkeitsprofil
        self.line_prof_cfd, = self.ax_prof.plot([], [], "b-o", markersize=4, label="CFD (LBM)")
        self.line_prof_bh, = self.ax_prof.plot([], [], "r--", linewidth=2, label="Burnham-Hallock")
        self.ax_prof.set_title("Tangentialgeschwindigkeit $v_\\theta(r)$ um Wirbelkern")
        self.ax_prof.set_xlabel("Radius $r$ [m]")
        self.ax_prof.set_ylabel("$v_\\theta$ [m/s]")
        self.ax_prof.grid(True, linestyle=":", alpha=0.6)
        self.ax_prof.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(self):
        """Aktualisiert alle 4 Subplots mit dem aktuellen Simulationsstand."""
        s = self.solver
        vort = s.get_vorticity_phys()
        ux_p, uy_p, u_mag = s.get_velocity_phys()

        # 1. Wirbelstärke & Kerne
        vmax_vort = max(float(np.max(np.abs(vort))), 1.0)
        self.im_vort.set_data(vort)
        self.im_vort.set_clim(-vmax_vort, vmax_vort)

        pos_l, pos_r = s.track_vortex_cores()
        self.scatter_l.set_data([pos_l[0]], [pos_l[1]])
        self.scatter_r.set_data([pos_r[0]], [pos_r[1]])

        # 2. Geschwindigkeit
        vmax_vel = max(float(np.max(u_mag)), 5.0)
        self.im_vel.set_data(u_mag)
        self.im_vel.set_clim(0, vmax_vel)

        # 3. Trajektorien
        times = np.array(s.history["time"])
        left_y = np.array(s.history["left_vortex_y"])
        right_y = np.array(s.history["right_vortex_y"])
        left_x = np.array(s.history["left_vortex_x"])
        right_x = np.array(s.history["right_vortex_x"])

        if len(times) > 0:
            mean_h = 0.5 * (left_y + right_y)
            sep = np.abs(right_x - left_x)
            self.line_traj_h.set_data(times, mean_h)
            self.line_traj_sep.set_data(times, sep)

            self.ax_traj.set_xlim(0, max(times[-1], 1.0))
            max_y_val = max(float(np.max(mean_h)), float(np.max(sep)), s.altitude_init) + 5.0
            self.ax_traj.set_ylim(0, max_y_val)

        # 4. Tangentialprofil
        radii = np.linspace(0.5, 0.35 * s.aircraft.wingspan, 25)
        v_cfd = compute_tangential_velocity_profile(
            ux_p, uy_p, s.x_mesh, s.y_mesh, center=pos_l, radii=radii
        )
        
        # Burnham-Hallock Referenzkurve
        gamma0 = s.gamma_0_phys
        rc = s.rc_phys
        v_bh = (gamma0 / (2.0 * np.pi)) * (radii / (radii ** 2 + rc ** 2))

        self.line_prof_cfd.set_data(radii, v_cfd)
        self.line_prof_bh.set_data(radii, v_bh)
        self.ax_prof.set_xlim(0, radii[-1])
        max_v_prof = max(float(np.max(v_cfd)), float(np.max(v_bh)), 1.0) * 1.15
        self.ax_prof.set_ylim(0, max_v_prof)

        self.fig.suptitle(
            f"Wirbelschleppen-Simulation - {s.aircraft.name} | t = {s.time_phys:.2f} s | Step: {s.current_step} | Höhe: {pos_l[1]:.1f} m",
            fontsize=13, fontweight="bold"
        )

        if self.interactive:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def run_interactive(self, total_steps: int = 500, steps_per_frame: int = 10):
        """Führt eine animierte Simulation im interaktiven Matplotlib-Fenster aus."""
        for step_idx in range(0, total_steps, steps_per_frame):
            self.solver.step(steps_per_frame)
            self.update()
        if self.interactive:
            plt.ioff()
            plt.show()


def plot_static_wake_summary(solver: CrossPlaneWakeSolver, output_path: Optional[str] = None) -> Figure:
    """
    Erstellt ein publikationsreifes Übersichtsdiagramm der Simulationsergebnisse.
    """
    plotter = LiveWakePlotter(solver, interactive=False)
    plotter.update()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plotter.fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return plotter.fig


def plot_validation_summary(val_results: Dict[str, Any], output_path: Optional[str] = None) -> Figure:
    """
    Erstellt einen 3-teiligen Validierungsberichtsgraph (Prandtl-Vergleich,
    Geschwindigkeitsprofil vs. Lamb-Oseen & Burnham-Hallock, Trajektorie im Bodeneffekt).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"CFD-Validierung gegen analytische Modelle ({val_results['aircraft']})", fontsize=14, fontweight="bold")

    # 1. Zirkulationsbalkendiagramm
    ax1 = axes[0]
    labels = ["Prandtl Lifting-Line", "Flugzeuggewicht\n(Kutta-Joukowski)"]
    values = [val_results["circulation_prandtl_m2_s"], val_results["circulation_weight_m2_s"]]
    bars = ax1.bar(labels, values, color=["#2b5c8f", "#d95f02"], width=0.5)
    ax1.set_ylabel(r"Initiale Zirkulation $\Gamma_0$ [$m^2/s$]")
    ax1.set_title(f"Zirkulations-Vergleich (Diff: {val_results['circulation_relative_diff_percent']:.2f}%)")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # 2. Tangentialprofil
    ax2 = axes[1]
    prof = val_results["profile_comparison"]
    ax2.plot(prof["radii"], prof["v_theta_cfd"], "b-o", label="CFD (LBM Solver)", markersize=4)
    ax2.plot(prof["radii"], prof["v_theta_burnham_hallock"], "r--", linewidth=2, label=f"Burnham-Hallock ($R^2={prof['r2_burnham_hallock']:.3f}$)")
    ax2.plot(prof["radii"], prof["v_theta_lamb_oseen"], "g:", linewidth=2, label=f"Lamb-Oseen ($R^2={prof['r2_lamb_oseen']:.3f}$)")
    ax2.set_xlabel("Radius $r$ [m]")
    ax2.set_ylabel(r"Tangentialgeschwindigkeit $v_\theta(r)$ [m/s]")
    ax2.set_title("Wirbelprofil-Validierung")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Trajektorie im Bodeneffekt
    ax3 = axes[2]
    traj = val_results["trajectory_comparison"]
    ax3.plot(traj["cfd_time"], traj["cfd_height"], "b-", linewidth=2.5, label="CFD (LBM Navier-Stokes)")
    ax3.plot(traj["analytical_time"], traj["analytical_height"], "r--", linewidth=2, label=f"Spiegelladungen (RMSE={traj['trajectory_rmse_m']:.2f} m)")
    ax3.set_xlabel("Zeit $t$ [s]")
    ax3.set_ylabel("Wirbelhöhe über Grund $h$ [m]")
    ax3.set_title("Absinken & Bodeneffekt-Trajektorie")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
