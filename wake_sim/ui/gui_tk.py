"""
Minimalistische und intuitive grafische Benutzeroberfläche (GUI) mit Tkinter.
"""

import sys
import os
from typing import Optional, Tuple, List, Dict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ..config import SimulationConfig, AircraftParameters, AIRCRAFT_PRESETS
from ..solver.cross_plane_wake import CrossPlaneWakeSolver
from ..solver.lbm_d2q9 import LBMD2Q9Solver
from ..validation.benchmark import run_analytical_validation
from ..visualization.live_plotter import LiveWakePlotter, plot_validation_summary
from ..postprocessing.metrics import compute_hazard_distance


class WakeSimGUI:
    """
    Hauptfenster der Tkinter-GUI für den Landungs-Wirbelschleppen-Simulator.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wirbelschleppen-Simulator (CFD / Lattice-Boltzmann)")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 700)

        # Simulationszustand
        self.is_running = False
        self.solver: Optional[CrossPlaneWakeSolver] = None
        self.config: Optional[SimulationConfig] = None
        self.anim_job = None
        self.steps_per_tick = 5

        self._setup_style()
        self._build_layout()
        self._on_preset_changed()
        self._reset_simulation()

    def _setup_style(self):
        """Konfiguriert das optische Design der Widgets."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("TButton", font=("Helvetica", 10, "bold"), padding=4)
        style.configure("Accent.TButton", foreground="white", background="#1a73e8")

    def _build_layout(self):
        """Erstellt die Aufteilung in Steuerungsleiste (links) und Plot-Leinwand (rechts)."""
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Linker Bereich: Parameter und Steuerung
        control_frame = ttk.Frame(main_pane, width=340, padding=8)
        main_pane.add(control_frame, weight=0)

        # Rechter Bereich: Matplotlib Canvas
        plot_frame = ttk.Frame(main_pane, padding=4)
        main_pane.add(plot_frame, weight=1)

        self._build_control_panel(control_frame)
        self._build_plot_panel(plot_frame)

    def _build_control_panel(self, parent: ttk.Frame):
        """Erstellt alle Steuer- und Eingabeelemente."""
        # Überschrift
        lbl_title = ttk.Label(parent, text="Flugzeug & Parameter", style="Header.TLabel")
        lbl_title.pack(anchor=tk.W, pady=(0, 6))

        # 1. Preset-Auswahl
        preset_frame = ttk.LabelFrame(parent, text="Flugzeug-Preset", padding=6)
        preset_frame.pack(fill=tk.X, pady=4)

        self.preset_var = tk.StringVar(value="A320")
        preset_combo = ttk.Combobox(
            preset_frame, textvariable=self.preset_var,
            values=["A320", "B737", "B777", "C172"], state="readonly"
        )
        preset_combo.pack(fill=tk.X, pady=2)
        preset_combo.bind("<<ComboboxSelected>>", lambda e: self._on_preset_changed())

        # 2. Physikalische Parameter
        param_frame = ttk.LabelFrame(parent, text="Landeanflug-Parameter", padding=6)
        param_frame.pack(fill=tk.X, pady=4)

        # Masse [kg]
        ttk.Label(param_frame, text="Masse [kg]:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.mass_var = tk.DoubleVar(value=66000.0)
        ttk.Entry(param_frame, textvariable=self.mass_var, width=10).grid(row=0, column=1, pady=2)

        # Spannweite [m]
        ttk.Label(param_frame, text="Spannweite b [m]:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.span_var = tk.DoubleVar(value=35.8)
        ttk.Entry(param_frame, textvariable=self.span_var, width=10).grid(row=1, column=1, pady=2)

        # Geschwindigkeit [m/s]
        ttk.Label(param_frame, text="Geschwindigkeit U_inf [m/s]:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.speed_var = tk.DoubleVar(value=70.0)
        ttk.Entry(param_frame, textvariable=self.speed_var, width=10).grid(row=2, column=1, pady=2)

        # Anfangshöhe [m]
        ttk.Label(param_frame, text="Anfangshöhe h_0 [m]:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.height_var = tk.DoubleVar(value=60.0)
        ttk.Entry(param_frame, textvariable=self.height_var, width=10).grid(row=3, column=1, pady=2)

        # Seitenwind [m/s]
        ttk.Label(param_frame, text="Seitenwind [m/s]:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.wind_var = tk.DoubleVar(value=0.0)
        ttk.Entry(param_frame, textvariable=self.wind_var, width=10).grid(row=4, column=1, pady=2)

        # 3. Gitter- und Solver-Einstellungen
        grid_frame = ttk.LabelFrame(parent, text="Numerisches Gitter (LBM)", padding=6)
        grid_frame.pack(fill=tk.X, pady=4)

        ttk.Label(grid_frame, text="Gitter NX x NY:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.grid_choice_var = tk.StringVar(value="200 x 140")
        grid_combo = ttk.Combobox(
            grid_frame, textvariable=self.grid_choice_var,
            values=["140 x 90 (Schnell)", "200 x 140 (Standard)", "280 x 180 (Fein)"],
            state="readonly", width=16
        )
        grid_combo.grid(row=0, column=1, pady=2)

        # 4. Telemetrie & ICAO-Info
        info_frame = ttk.LabelFrame(parent, text="Wirbelschleppen-Kenngrößen", padding=6)
        info_frame.pack(fill=tk.X, pady=4)

        self.lbl_circ = ttk.Label(info_frame, text="Zirkulation Gamma_0: -- m^2/s")
        self.lbl_circ.pack(anchor=tk.W, pady=1)

        self.lbl_w0 = ttk.Label(info_frame, text="Absinkrate w_0: -- m/s")
        self.lbl_w0.pack(anchor=tk.W, pady=1)

        self.lbl_icao = ttk.Label(info_frame, text="ICAO Staffelung: -- NM")
        self.lbl_icao.pack(anchor=tk.W, pady=1)

        self.lbl_status = ttk.Label(info_frame, text="Status: Bereit", foreground="#2b5c8f", font=("Helvetica", 10, "bold"))
        self.lbl_status.pack(anchor=tk.W, pady=3)

        # 5. Steuerungs-Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        self.btn_start = ttk.Button(btn_frame, text="Start / Pause", command=self._toggle_simulation)
        self.btn_start.pack(fill=tk.X, pady=3)

        self.btn_step = ttk.Button(btn_frame, text="Einzelschritt (+10)", command=lambda: self._step_once(10))
        self.btn_step.pack(fill=tk.X, pady=3)

        self.btn_reset = ttk.Button(btn_frame, text="Zurücksetzen", command=self._reset_simulation)
        self.btn_reset.pack(fill=tk.X, pady=3)

        ttk.Separator(btn_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        self.btn_validate = ttk.Button(btn_frame, text="Analytische Validierung", command=self._run_validation)
        self.btn_validate.pack(fill=tk.X, pady=3)

        self.btn_export = ttk.Button(btn_frame, text="Diagramm speichern (PNG)", command=self._export_plot)
        self.btn_export.pack(fill=tk.X, pady=3)

    def _build_plot_panel(self, parent: ttk.Frame):
        """Erstellt die Matplotlib-Leinwand."""
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()

    def _on_preset_changed(self):
        """Aktualisiert Eingabefelder anhand des gewählten Flugzeugmusters."""
        p_name = self.preset_var.get()
        if p_name in AIRCRAFT_PRESETS:
            ac = AIRCRAFT_PRESETS[p_name]
            self.mass_var.set(ac.mass)
            self.span_var.set(ac.wingspan)
            self.speed_var.set(ac.approach_speed)
            self.height_var.set(min(60.0, ac.wingspan * 1.8))
            self._update_telemetry_labels(ac)

    def _update_telemetry_labels(self, ac: AircraftParameters):
        """Aktualisiert Zirkulation, Absinkrate und ICAO-Staffelung."""
        gamma0 = ac.initial_circulation()
        w0 = ac.initial_descent_rate()
        hazard = compute_hazard_distance(ac, follower_category="Medium")

        self.lbl_circ.config(text=f"Zirkulation Gamma_0: {gamma0:.1f} m^2/s")
        self.lbl_w0.config(text=f"Absinkrate w_0: {w0:.2f} m/s")
        self.lbl_icao.config(text=f"ICAO Staffelung: {hazard['separation_nm']:.1f} NM ({hazard['separation_m']:,.0f} m)")

    def _parse_grid_choice(self) -> Tuple[int, int]:
        """Liest NX und NY aus der Auswahlliste."""
        choice = self.grid_choice_var.get()
        if "140" in choice:
            return 140, 90
        elif "280" in choice:
            return 280, 180
        return 200, 140

    def _reset_simulation(self):
        """Initialisiert den Solver und die Plots neu."""
        if self.is_running:
            self._toggle_simulation()

        nx, ny = self._parse_grid_choice()
        ac = AircraftParameters(
            name=self.preset_var.get(),
            mass=self.mass_var.get(),
            wingspan=self.span_var.get(),
            approach_speed=self.speed_var.get()
        )
        self._update_telemetry_labels(ac)

        self.config = SimulationConfig(
            aircraft_name=ac.name,
            custom_aircraft=ac,
            initial_altitude=self.height_var.get(),
            crosswind=self.wind_var.get(),
            nx=nx,
            ny=ny,
            domain_width_m=ac.wingspan * 3.5,
            domain_height_m=ac.wingspan * 2.2
        )

        self.solver = CrossPlaneWakeSolver(self.config, aircraft=ac)
        self.lbl_status.config(text=f"Status: Bereit (t = 0.0s)")
        self._draw_initial_plots()

    def _draw_initial_plots(self):
        """Zeichnet die 4 Subplots neu."""
        self.fig.clear()
        self.axes = self.fig.subplots(2, 2)
        s = self.solver
        vort = s.get_vorticity_phys()
        _, _, u_mag = s.get_velocity_phys()

        # 1. Wirbelstärke
        vmax_vort = max(float(np.max(np.abs(vort))), 1.0)
        self.im_vort = self.axes[0, 0].imshow(
            vort, extent=[s.x_phys_1d[0], s.x_phys_1d[-1], s.y_phys_1d[0], s.y_phys_1d[-1]],
            origin="lower", cmap="seismic", vmin=-vmax_vort, vmax=vmax_vort, aspect="auto"
        )
        self.axes[0, 0].axhline(0, color="k", linewidth=2.0)
        self.axes[0, 0].set_title("Wirbelstärke $\\omega_z$ [1/s]")
        self.axes[0, 0].set_ylabel("Höhe $y$ [m]")

        # 2. Geschwindigkeit
        vmax_vel = max(float(np.max(u_mag)), 5.0)
        self.im_vel = self.axes[0, 1].imshow(
            u_mag, extent=[s.x_phys_1d[0], s.x_phys_1d[-1], s.y_phys_1d[0], s.y_phys_1d[-1]],
            origin="lower", cmap="plasma", vmin=0, vmax=vmax_vel, aspect="auto"
        )
        self.axes[0, 1].set_title(r"Geschwindigkeitsbetrag $|\mathbf{u}|$ [m/s]")

        # 3. Trajektorie
        self.line_h, = self.axes[1, 0].plot([], [], "b-", linewidth=2, label="Höhe h(t)")
        self.axes[1, 0].set_title("Wirbelhöhe im Bodeneffekt")
        self.axes[1, 0].set_xlabel("Zeit [s]")
        self.axes[1, 0].set_ylabel("Höhe [m]")
        self.axes[1, 0].grid(True, linestyle=":", alpha=0.6)

        # 4. Kinetische Energie
        self.line_e, = self.axes[1, 1].plot([], [], "r-", linewidth=2, label="Energie E(t)")
        self.axes[1, 1].set_title("Kinetische Energie")
        self.axes[1, 1].set_xlabel("Zeit [s]")
        self.axes[1, 1].set_ylabel("Energie [J]")
        self.axes[1, 1].grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

    def _toggle_simulation(self):
        """Startet oder pausiert die Simulation."""
        self.is_running = not self.is_running
        if self.is_running:
            self.lbl_status.config(text="Status: Simulation läuft...")
            self._simulation_tick()
        else:
            self.lbl_status.config(text=f"Status: Pausiert (t = {self.solver.time_phys:.2f}s)")
            if self.anim_job:
                self.root.after_cancel(self.anim_job)
                self.anim_job = None

    def _step_once(self, n_steps: int = 10):
        """Führt eine feste Anzahl von Schritten aus und aktualisiert die Ansicht."""
        if self.solver is not None:
            self.solver.step(n_steps)
            self._update_plot_data()

    def _simulation_tick(self):
        """Periodische Animationsschleife."""
        if not self.is_running:
            return

        self.solver.step(self.steps_per_tick)
        self._update_plot_data()

        # Nächsten Tick anfordern (~30 FPS)
        self.anim_job = self.root.after(30, self._simulation_tick)

    def _update_plot_data(self):
        """Aktualisiert die Bilddaten der Subplots."""
        s = self.solver
        vort = s.get_vorticity_phys()
        _, _, u_mag = s.get_velocity_phys()

        vmax_vort = max(float(np.max(np.abs(vort))), 1.0)
        self.im_vort.set_data(vort)
        self.im_vort.set_clim(-vmax_vort, vmax_vort)

        vmax_vel = max(float(np.max(u_mag)), 5.0)
        self.im_vel.set_data(u_mag)
        self.im_vel.set_clim(0, vmax_vel)

        times = np.array(s.history["time"])
        left_y = np.array(s.history["left_vortex_y"])
        right_y = np.array(s.history["right_vortex_y"])
        e_kin = np.array(s.history["kinetic_energy"])

        if len(times) > 0:
            mean_h = 0.5 * (left_y + right_y)
            self.line_h.set_data(times, mean_h)
            self.axes[1, 0].set_xlim(0, max(times[-1], 1.0))
            self.axes[1, 0].set_ylim(0, max(float(np.max(mean_h)), s.altitude_init) + 5.0)

            self.line_e.set_data(times, e_kin)
            self.axes[1, 1].set_xlim(0, max(times[-1], 1.0))
            self.axes[1, 1].set_ylim(0, max(float(np.max(e_kin)), 1.0) * 1.1)

            pos_l, _ = s.track_vortex_cores()
            self.lbl_status.config(text=f"t = {s.time_phys:.2f} s | Step {s.current_step} | Höhe: {pos_l[1]:.1f} m")

        self.canvas.draw_idle()

    def _run_validation(self):
        """Führt die analytische Validierung aus und zeigt den Ergebnisgraph."""
        if self.is_running:
            self._toggle_simulation()

        p_name = self.preset_var.get()
        self.lbl_status.config(text="Führe Validierung durch...")
        self.root.update()

        val = run_analytical_validation(p_name, sim_steps=120)
        
        # Validierungsfenster
        val_win = tk.Toplevel(self.root)
        val_win.title(f"Validierungsbericht - {val['aircraft']}")
        val_win.geometry("960x480")

        fig_val = plot_validation_summary(val)
        canvas_val = FigureCanvasTkAgg(fig_val, master=val_win)
        canvas_val.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_val.draw()

        self.lbl_status.config(text=f"Validierung abgeschlossen [{val['status']}]")

    def _export_plot(self):
        """Speichert die aktuelle Abbildung als PNG."""
        fpath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")]
        )
        if fpath:
            self.fig.savefig(fpath, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Export erfolgreich", f"Grafik gespeichert unter:\n{fpath}")


def run_gui():
    """Startet die Tkinter GUI-Schleife."""
    root = tk.Tk()
    app = WakeSimGUI(root)
    root.mainloop()
