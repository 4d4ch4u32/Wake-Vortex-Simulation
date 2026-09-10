"""
Kommandozeilenschnittstelle (CLI) für die Wirbelschleppen-CFD-Simulation.
"""

import sys
import argparse
import time
import numpy as np

from ..config import SimulationConfig, AircraftParameters, AIRCRAFT_PRESETS
from ..solver.cross_plane_wake import CrossPlaneWakeSolver
from ..solver.lbm_d2q9 import LBMD2Q9Solver
from ..solver.lbm_d3q19 import LBMD3Q19Solver
from ..validation.benchmark import run_analytical_validation
from ..visualization.live_plotter import LiveWakePlotter, plot_static_wake_summary, plot_validation_summary
from ..visualization.export import export_simulation_results, export_animation_gif
from ..postprocessing.metrics import compute_hazard_distance, compute_energy_decay


def parse_arguments(args=None):
    """Parst die CLI-Argumente."""
    parser = argparse.ArgumentParser(
        description="2D/3D CFD-Simulation von Flugzeug-Wirbelschleppen während der Landung (LBM Navier-Stokes)."
    )
    
    parser.add_argument(
        "--mode", choices=["cross_plane", "airfoil", "3d", "validate", "benchmark"],
        default="cross_plane",
        help="Simulationsmodus: 'cross_plane' (Landung/Bodeneffekt), 'airfoil' (NACA 0012 Profil), '3d' (3D-Flügel), 'validate' (Vergleich mit analytischen Modellen)"
    )
    parser.add_argument(
        "--aircraft", choices=list(AIRCRAFT_PRESETS.keys()),
        default="A320",
        help="Flugzeugmuster-Preset: A320 (Airbus A320), B737 (Boeing 737-800), B777 (Boeing 777-300ER), C172 (Cessna 172)"
    )
    parser.add_argument(
        "--profile", type=str, default="0012",
        help="4-stelliger NACA Profilcode (z. B. '0012', '2412', '4412')"
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help="Landegeschwindigkeit U_inf in m/s (Standard: Preset-Wert, z. B. 70 m/s für A320)"
    )
    parser.add_argument(
        "--height", type=float, default=60.0,
        help="Anfängliche Wirbelhöhe über der Landebahn h_0 in m (Standard: 60.0)"
    )
    parser.add_argument(
        "--aoa", type=float, default=None,
        help="Anstellwinkel alpha in Grad (Standard: Preset-Wert)"
    )
    parser.add_argument(
        "--crosswind", type=float, default=0.0,
        help="Seitenwind-Geschwindigkeit u_cross in m/s (Standard: 0.0)"
    )
    parser.add_argument(
        "--steps", type=int, default=300,
        help="Gesamtzahl der Simulationsschritte (Standard: 300)"
    )
    parser.add_argument(
        "--nx", type=int, default=240,
        help="Gitterauflösung in X-Richtung (Standard: 240)"
    )
    parser.add_argument(
        "--ny", type=int, default=160,
        help="Gitterauflösung in Y-Richtung (Standard: 160)"
    )
    parser.add_argument(
        "--nz", type=int, default=40,
        help="Gitterauflösung in Z-Richtung für 3D-Modus (Standard: 40)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Interaktive Echtzeit-Animation während der Simulation anzeigen"
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Dateipfad zum Speichern des Ergebnisdiagramms (z. B. 'output/wake_result.png')"
    )
    parser.add_argument(
        "--save-csv", type=str, default=None,
        help="Dateipfad zum Exportieren der Wirbeltrajektorie als CSV (z. B. 'output/wake_data.csv')"
    )
    parser.add_argument(
        "--save-gif", type=str, default=None,
        help="Dateipfad zum Exportieren einer animierten GIF-Datei (z. B. 'output/wake_anim.gif')"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Grafische Benutzeroberfläche (Tkinter GUI) starten"
    )

    return parser.parse_args(args)


def run_cli(args=None):
    """Haupteinstiegspunkt für CLI-Ausführung."""
    parsed = parse_arguments(args)

    if parsed.gui:
        from .gui_tk import run_gui
        run_gui()
        return

    print("=" * 72)
    print("  CFD-WIRBELSCHLEPPEN-SIMULATOR (LATTICE-BOLTZMANN-METHODE)")
    print("=" * 72)

    # Validierungs-Modus
    if parsed.mode in ["validate", "benchmark"]:
        print(f"[*] Starte analytische Validierung für Flugzeugmuster: {parsed.aircraft}...")
        t0 = time.time()
        val = run_analytical_validation(parsed.aircraft, sim_steps=parsed.steps)
        dt_calc = time.time() - t0

        print(f"\n--- VALIDIERUNGSERGEBNIS ({val['aircraft']}) in {dt_calc:.2f}s ---")
        print(f"  Flugzeugmasse:                  {val['mass_kg']:,.0f} kg")
        print(f"  Spannweite:                     {val['wingspan_m']:.1f} m")
        print(f"  Landegeschwindigkeit:           {val['approach_speed_m_s']:.1f} m/s")
        print(f"  Zirkulation (Prandtl):          {val['circulation_prandtl_m2_s']:.2f} m^2/s")
        print(f"  Zirkulation (Gewichtskraft):    {val['circulation_weight_m2_s']:.2f} m^2/s")
        print(f"  Relative Zirkulationsdifferenz: {val['circulation_relative_diff_percent']:.2f} %")
        print(f"  Initiale Absinkgeschwindigkeit: {val['initial_descent_rate_analytical_m_s']:.2f} m/s")
        
        prof = val["profile_comparison"]
        print(f"\n  Profil-Vergleich mit Burnham-Hallock: R^2 = {prof['r2_burnham_hallock']:.4f} | RMSE = {prof['rmse_burnham_hallock_m_s']:.3f} m/s")
        print(f"  Profil-Vergleich mit Lamb-Oseen:      R^2 = {prof['r2_lamb_oseen']:.4f} | RMSE = {prof['rmse_lamb_oseen_m_s']:.3f} m/s")
        
        traj = val["trajectory_comparison"]
        print(f"  Trajektorien-RMSE (Spiegelladungen):  {traj['trajectory_rmse_m']:.3f} m")
        print(f"  Status:                               [{val['status']}]")

        if parsed.save_plot:
            plot_validation_summary(val, output_path=parsed.save_plot)
            print(f"[+] Validierungsdiagramm gespeichert: {parsed.save_plot}")
        elif parsed.visualize:
            fig = plot_validation_summary(val)
            import matplotlib.pyplot as plt
            plt.show()
        return

    # Flugzeugkonfiguration vorbereiten
    aircraft = AIRCRAFT_PRESETS.get(parsed.aircraft, AIRCRAFT_PRESETS["A320"])
    if parsed.speed is not None:
        aircraft.approach_speed = parsed.speed
    if parsed.aoa is not None:
        aircraft.angle_of_attack_deg = parsed.aoa
    if parsed.profile is not None:
        aircraft.airfoil = f"NACA{parsed.profile}"

    config = SimulationConfig(
        mode=parsed.mode,
        aircraft_name=parsed.aircraft,
        custom_aircraft=aircraft,
        initial_altitude=parsed.height,
        crosswind=parsed.crosswind,
        nx=parsed.nx,
        ny=parsed.ny,
        nz=parsed.nz,
        total_steps=parsed.steps
    )

    print(f"  Flugzeug:      {aircraft.name} (MTOW: {aircraft.mass:,.0f} kg, b = {aircraft.wingspan:.1f} m)")
    print(f"  Landeanflug:   U_inf = {aircraft.approach_speed:.1f} m/s, h_0 = {config.initial_altitude:.1f} m, alpha = {aircraft.angle_of_attack_deg:.1f}°")
    print(f"  Modus:         {parsed.mode.upper()} | Gitter: {config.nx}x{config.ny}" + (f"x{config.nz}" if parsed.mode == '3d' else ""))
    print(f"  Wirbelparameter: Gamma_0 = {aircraft.initial_circulation():.1f} m^2/s, b_0 = {aircraft.vortex_spacing:.1f} m, w_0 = {aircraft.initial_descent_rate():.2f} m/s")
    
    hazard = compute_hazard_distance(aircraft, follower_category="Medium")
    print(f"  ICAO-Staffelung: {hazard['separation_nm']:.1f} NM ({hazard['separation_m']:,.0f} m, ca. {hazard['separation_time_s']:.0f} s)")
    print("-" * 72)

    # Solver instanziieren und ausführen
    if parsed.mode == "cross_plane":
        solver = CrossPlaneWakeSolver(config, aircraft=aircraft)
        print(f"[*] Starte Cross-Plane Wirbelschleppen-Simulation ({parsed.steps} Zeitschritte)...")
        
        t0 = time.time()
        if parsed.visualize:
            plotter = LiveWakePlotter(solver, interactive=True)
            plotter.run_interactive(total_steps=parsed.steps, steps_per_frame=max(1, parsed.steps // 50))
        else:
            solver.step(parsed.steps)
        t_calc = time.time() - t0

        pos_l, pos_r = solver.track_vortex_cores()
        print(f"\n[+] Simulation erfolgreich abgeschlossen in {t_calc:.2f} s ({parsed.steps / max(t_calc, 1e-4):.1f} Steps/s).")
        print(f"  Endzeit:                 t = {solver.time_phys:.2f} s")
        print(f"  Endposition Linker Wirbel:  (x={pos_l[0]:.1f} m, y={pos_l[1]:.1f} m)")
        print(f"  Endposition Rechter Wirbel: (x={pos_r[0]:.1f} m, y={pos_r[1]:.1f} m)")
        print(f"  Wirbelabstand am Ende:   b = {abs(pos_r[0] - pos_l[0]):.1f} m")
        
        rebound = solver.history["left_vortex_y"]
        if len(rebound) > 0:
            min_h = min(rebound)
            print(f"  Minimale Wirbelhöhe:     h_min = {min_h:.1f} m")

        if parsed.save_plot:
            plot_static_wake_summary(solver, output_path=parsed.save_plot)
            print(f"[+] Diagramm gespeichert: {parsed.save_plot}")

        if parsed.save_csv:
            export_simulation_results(solver, parsed.save_csv)
            print(f"[+] CSV-Daten exportiert: {parsed.save_csv}")

        if parsed.save_gif:
            print(f"[*] Erzeuge animiertes GIF ({parsed.save_gif})...")
            # Frischen Solver für GIF-Aufzeichnung erstellen
            gif_solver = CrossPlaneWakeSolver(config, aircraft=aircraft)
            export_animation_gif(gif_solver, parsed.save_gif, n_frames=30, steps_per_frame=parsed.steps // 30)
            print(f"[+] GIF erfolgreich exportiert: {parsed.save_gif}")

    elif parsed.mode == "airfoil":
        solver = LBMD2Q9Solver(
            config,
            airfoil_code=parsed.profile,
            chord_length_m=aircraft.chord_length,
            angle_of_attack_deg=aircraft.angle_of_attack_deg,
            inflow_speed_m_s=aircraft.approach_speed
        )
        print(f"[*] Starte NACA {parsed.profile} Profilströmungssimulation ({parsed.steps} Schritte)...")
        t0 = time.time()
        solver.step(parsed.steps)
        t_calc = time.time() - t0
        
        _, _, cl, cd = solver.compute_aerodynamic_forces()
        print(f"\n[+] Profilsimulation abgeschlossen in {t_calc:.2f} s:")
        print(f"  Auftriebsbeiwert CL:     {cl:.4f}")
        print(f"  Widerstandsbeiwert CD:   {cd:.4f}")
        print(f"  Gleitzahl CL/CD:         {cl / max(cd, 1e-4):.2f}")

    elif parsed.mode == "3d":
        solver = LBMD3Q19Solver(
            config,
            wingspan_m=min(aircraft.wingspan, 20.0),
            chord_length_m=aircraft.chord_length,
            inflow_speed_m_s=aircraft.approach_speed
        )
        print(f"[*] Starte 3D D3Q19 Flügel-Simulation ({parsed.steps} Schritte)...")
        t0 = time.time()
        solver.step(parsed.steps)
        t_calc = time.time() - t0
        print(f"\n[+] 3D-Simulation abgeschlossen in {t_calc:.2f} s.")
