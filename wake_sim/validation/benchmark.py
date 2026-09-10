"""
Automatisierte Validierungsroutinen und Fehlermetriken gegen analytische Theorien.
"""

from typing import Dict, Any, Tuple
import numpy as np

from ..config import AircraftParameters, SimulationConfig, AIRCRAFT_PRESETS
from ..solver.cross_plane_wake import CrossPlaneWakeSolver
from ..postprocessing.aerodynamics import compute_tangential_velocity_profile, compute_circulation_profile
from .prandtl_lifting_line import PrandtlLiftingLine
from .vortex_models import LambOseenVortex, BurnhamHallockVortex, GroundEffectImageModel


def run_analytical_validation(
    aircraft_name: str = "A320",
    sim_steps: int = 150
) -> Dict[str, Any]:
    """
    Führt einen vollständigen Vergleich zwischen der LBM-CFD-Simulation und den
    analytischen Lösungen (Prandtl, Burnham-Hallock, Lamb-Oseen, Spiegelladungen) durch.

    :param aircraft_name: Preset-Name des Flugzeugs (z. B. 'A320', 'B737', 'B777', 'C172')
    :param sim_steps: Anzahl der CFD-Zeitschritte für den Vergleich
    :return: Ausführliches Dictionary mit Validierungsergebnissen und Fehlermetriken
    """
    aircraft = AIRCRAFT_PRESETS.get(aircraft_name, AIRCRAFT_PRESETS["A320"])
    
    # 1. Analytische Prandtl-Berechnung
    prandtl = PrandtlLiftingLine(
        wingspan=aircraft.wingspan,
        wing_area=aircraft.wing_area,
        velocity=aircraft.approach_speed,
        air_density=1.225
    )
    # Erforderlicher Auftriebsbeiwert im stationären Landeanflug (L = W)
    q_inf = 0.5 * 1.225 * (aircraft.approach_speed ** 2)
    cl_required = aircraft.weight / (q_inf * aircraft.wing_area)
    
    # Zirkulation nach Prandtl's elliptischer Auftriebsverteilung
    gamma0_prandtl = prandtl.root_circulation(cl_required)
    gamma0_weight = aircraft.initial_circulation()
    w0_analytical = aircraft.initial_descent_rate()
    
    # 2. CFD-Simulation aufsetzen und rechnen
    config = SimulationConfig(
        aircraft_name=aircraft_name,
        initial_altitude=min(60.0, aircraft.wingspan * 1.8),
        nx=160,
        ny=120,
        domain_width_m=aircraft.wingspan * 3.5,
        domain_height_m=aircraft.wingspan * 2.2,
        total_steps=sim_steps
    )
    solver = CrossPlaneWakeSolver(config, aircraft=aircraft)
    solver.step(sim_steps)
    
    # 3. Tangentialgeschwindigkeitsprofil vergleichen
    pos_l, _ = solver.track_vortex_cores()
    radii = np.linspace(0.5, 0.4 * aircraft.wingspan, 40)
    ux_p, uy_p, _ = solver.get_velocity_phys()
    
    v_theta_cfd = compute_tangential_velocity_profile(
        ux_p, uy_p, solver.x_mesh, solver.y_mesh,
        center=pos_l, radii=radii
    )
    
    # Analytische Modelle auswerten
    bh_model = BurnhamHallockVortex(gamma_0=gamma0_weight, core_radius=aircraft.initial_core_radius())
    lo_model = LambOseenVortex(gamma_0=gamma0_weight, core_radius_0=aircraft.initial_core_radius())
    
    v_theta_bh = bh_model.tangential_velocity(radii)
    v_theta_lo = lo_model.tangential_velocity(radii, time=solver.time_phys)
    
    # Fehlermetriken (RMSE und R^2)
    rmse_bh = float(np.sqrt(np.mean((v_theta_cfd - v_theta_bh) ** 2)))
    rmse_lo = float(np.sqrt(np.mean((v_theta_cfd - v_theta_lo) ** 2)))
    
    ss_tot = np.sum((v_theta_cfd - np.mean(v_theta_cfd)) ** 2)
    r2_bh = float(1.0 - np.sum((v_theta_cfd - v_theta_bh) ** 2) / max(ss_tot, 1e-6))
    r2_lo = float(1.0 - np.sum((v_theta_cfd - v_theta_lo) ** 2) / max(ss_tot, 1e-6))
    
    # 4. Trajektorien-Vergleich mit Spiegelladungsmodell
    image_model = GroundEffectImageModel(
        gamma_0=gamma0_weight,
        initial_spacing=aircraft.vortex_spacing,
        initial_altitude=config.initial_altitude
    )
    traj_analytical = image_model.compute_trajectory(
        t_max=solver.time_phys,
        dt=max(solver.dt_phys, 0.01)
    )
    
    cfd_times = np.array(solver.history["time"])
    cfd_heights = 0.5 * (np.array(solver.history["left_vortex_y"]) + np.array(solver.history["right_vortex_y"]))
    
    # Interpolation der analytischen Höhen auf CFD-Zeitschritte
    analytical_heights_interp = np.interp(cfd_times, traj_analytical["time"], traj_analytical["height"])
    traj_rmse = float(np.sqrt(np.mean((cfd_heights - analytical_heights_interp) ** 2)))
    
    # Zusammenfassung
    return {
        "aircraft": aircraft.name,
        "mass_kg": aircraft.mass,
        "wingspan_m": aircraft.wingspan,
        "approach_speed_m_s": aircraft.approach_speed,
        "circulation_prandtl_m2_s": gamma0_prandtl,
        "circulation_weight_m2_s": gamma0_weight,
        "circulation_relative_diff_percent": float(abs(gamma0_prandtl - gamma0_weight) / gamma0_weight * 100.0),
        "initial_descent_rate_analytical_m_s": w0_analytical,
        "profile_comparison": {
            "radii": radii,
            "v_theta_cfd": v_theta_cfd,
            "v_theta_burnham_hallock": v_theta_bh,
            "v_theta_lamb_oseen": v_theta_lo,
            "rmse_burnham_hallock_m_s": rmse_bh,
            "rmse_lamb_oseen_m_s": rmse_lo,
            "r2_burnham_hallock": r2_bh,
            "r2_lamb_oseen": r2_lo
        },
        "trajectory_comparison": {
            "cfd_time": cfd_times,
            "cfd_height": cfd_heights,
            "analytical_time": traj_analytical["time"],
            "analytical_height": traj_analytical["height"],
            "trajectory_rmse_m": traj_rmse
        },
        "status": "PASSED" if r2_bh > 0.75 or r2_lo > 0.75 else "WARNING"
    }
