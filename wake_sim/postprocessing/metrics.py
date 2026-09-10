"""
Metriken zur Wirbelschleppen-Gefahrenanalyse und Energieerhaltung.
"""

from typing import Dict, Tuple
import numpy as np
from ..config import AircraftParameters


# ICAO Staffelungsempfehlungen für Landeanflüge (in Nautischen Meilen [NM])
# Leader-Kategorie -> Follower-Kategorie: [Heavy, Medium, Light]
ICAO_SEPARATION_MATRIX_NM = {
    "Heavy": {"Heavy": 4.0, "Medium": 5.0, "Light": 6.0},
    "Medium": {"Heavy": 3.0, "Medium": 3.0, "Light": 5.0},
    "Light": {"Heavy": 3.0, "Medium": 3.0, "Light": 3.0},
}


def get_icao_category(aircraft: AircraftParameters) -> str:
    """Bestimmt die ICAO-Wirbelschleppenkategorie nach MTOW."""
    # Heavy: > 136.000 kg, Medium: 7.000 - 136.000 kg, Light: < 7.000 kg
    if aircraft.mass >= 136000.0:
        return "Heavy"
    elif aircraft.mass >= 7000.0:
        return "Medium"
    else:
        return "Light"


def compute_hazard_distance(
    leader: AircraftParameters,
    follower_category: str = "Medium"
) -> Dict[str, float]:
    """
    Berechnet die empfohlene ICAO-Staffelungsdistanz und -zeit bei der Landung.
    """
    leader_cat = get_icao_category(leader)
    sep_nm = ICAO_SEPARATION_MATRIX_NM.get(leader_cat, {}).get(follower_category, 5.0)
    
    # Umrechnung in Meter und Sekunden
    sep_meters = sep_nm * 1852.0
    sep_seconds = sep_meters / max(leader.approach_speed, 1.0)
    
    return {
        "leader_category": leader_cat,
        "follower_category": follower_category,
        "separation_nm": sep_nm,
        "separation_m": sep_meters,
        "separation_time_s": sep_seconds
    }


def compute_energy_decay(kinetic_energy_history: np.ndarray) -> Dict[str, float]:
    """
    Berechnet die relative Energiedissipation der Wirbel.
    """
    if len(kinetic_energy_history) == 0:
        return {"initial_energy_j": 0.0, "final_energy_j": 0.0, "decay_ratio": 0.0}

    e_init = float(kinetic_energy_history[0])
    e_final = float(kinetic_energy_history[-1])
    decay_ratio = 1.0 - (e_final / max(e_init, 1e-9))

    return {
        "initial_energy_j": e_init,
        "final_energy_j": e_final,
        "decay_ratio": max(0.0, min(1.0, decay_ratio))
    }
