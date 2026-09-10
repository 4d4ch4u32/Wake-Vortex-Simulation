"""
Physikalische Konstanten, Flugzeugparameter und Simulationskonfiguration.
"""

from dataclasses import dataclass, field
import numpy as np


# Standard-Atmosphäre auf Meereshöhe (ISA)
AIR_DENSITY_DEFAULT: float = 1.225       # Luftdichte rho [kg/m^3]
AIR_KINEMATIC_VISCOSITY: float = 1.5e-5  # Kinematische Viskosität nu [m^2/s]
GRAVITY_ACCELERATION: float = 9.81       # Erdbeschleunigung g [m/s^2]
SPEED_OF_SOUND_AIR: float = 340.0        # Schallgeschwindigkeit Luft [m/s]


@dataclass
class AircraftParameters:
    """
    Geometrische und aerodynamische Kenndaten eines Flugzeugs.
    """
    name: str = "Airbus A320"
    mass: float = 66000.0                # Gesamtmasse M [kg] (typisches Landegewicht)
    wingspan: float = 35.8               # Spannweite b [m]
    wing_area: float = 122.6             # Flügelfläche S [m^2]
    chord_length: float = 4.19           # Mittlere Flügeltiefe c [m]
    approach_speed: float = 70.0         # Landegeschwindigkeit U_inf [m/s] (~136 kts)
    airfoil: str = "NACA0012"            # Standardprofil (z.B. NACA 0012)
    angle_of_attack_deg: float = 5.0     # Anstellwinkel alpha [Grad]
    glide_slope_deg: float = 3.0         # Gleitpfadwinkel gamma [Grad]

    @property
    def weight(self) -> float:
        """Gewichtskraft W = M * g [N]."""
        return self.mass * GRAVITY_ACCELERATION

    @property
    def aspect_ratio(self) -> float:
        """Flügelstreckung Lambda = b^2 / S [-]."""
        return (self.wingspan ** 2) / self.wing_area

    @property
    def vortex_spacing(self) -> float:
        """
        Initialer Wirbelabstand b_0 nach elliptischer Auftriebsverteilung (Prandtl).
        b_0 = (pi / 4) * b approx 0.7854 * b.
        """
        return (np.pi / 4.0) * self.wingspan

    def initial_circulation(self, density: float = AIR_DENSITY_DEFAULT) -> float:
        """
        Berechnet die initiale Wirbelzirkulation Gamma_0 nach der Kutta-Joukowski /
        Prandtl-Traglinientheorie:
        Gamma_0 = W / (rho * U_inf * b_0) = (4 * W) / (pi * rho * U_inf * b) [m^2/s].
        """
        return self.weight / (density * self.approach_speed * self.vortex_spacing)

    def initial_descent_rate(self, density: float = AIR_DENSITY_DEFAULT) -> float:
        """
        Berechnet die initiale gegenseitige Absinkgeschwindigkeit der Wirbel:
        w_0 = Gamma_0 / (2 * pi * b_0) [m/s].
        """
        gamma_0 = self.initial_circulation(density)
        return gamma_0 / (2.0 * np.pi * self.vortex_spacing)

    def initial_core_radius(self) -> float:
        """
        Typischer Wirbelkernradius r_c (empirisch ca. 3.5% bis 5% der Spannweite).
        """
        return 0.04 * self.wingspan


# Vordefinierte Flugzeugmuster
AIRCRAFT_PRESETS = {
    "A320": AircraftParameters(
        name="Airbus A320-200",
        mass=66000.0,
        wingspan=35.8,
        wing_area=122.6,
        chord_length=4.19,
        approach_speed=70.0,
        airfoil="NACA0012",
        angle_of_attack_deg=5.0,
        glide_slope_deg=3.0
    ),
    "B737": AircraftParameters(
        name="Boeing 737-800",
        mass=65000.0,
        wingspan=35.8,
        wing_area=124.6,
        chord_length=4.17,
        approach_speed=72.0,
        airfoil="NACA0012",
        angle_of_attack_deg=4.5,
        glide_slope_deg=3.0
    ),
    "B777": AircraftParameters(
        name="Boeing 777-300ER (Heavy)",
        mass=250000.0,
        wingspan=64.8,
        wing_area=436.8,
        chord_length=7.50,
        approach_speed=75.0,
        airfoil="NACA0012",
        angle_of_attack_deg=5.5,
        glide_slope_deg=3.0
    ),
    "C172": AircraftParameters(
        name="Cessna 172 Skyhawk (Light)",
        mass=1100.0,
        wingspan=11.0,
        wing_area=16.2,
        chord_length=1.47,
        approach_speed=30.0,
        airfoil="NACA2412",
        angle_of_attack_deg=4.0,
        glide_slope_deg=3.0
    ),
}


@dataclass
class SimulationConfig:
    """
    Konfiguration der numerischen Strömungssimulation.
    """
    # Physikalische Domäne (in SI-Einheiten [m] bzw. [s])
    mode: str = "cross_plane"           # "cross_plane" (Landung/Bodeneffekt), "airfoil" (2D-Profil) oder "3d"
    aircraft_name: str = "A320"         # Flugzeug-Preset
    custom_aircraft: AircraftParameters = field(default_factory=lambda: AIRCRAFT_PRESETS["A320"])
    
    # Physikalische Parameter
    density: float = AIR_DENSITY_DEFAULT
    kinematic_viscosity: float = AIR_KINEMATIC_VISCOSITY
    initial_altitude: float = 60.0       # Anfangshöhe der Wirbel über Grund h_0 [m]
    crosswind: float = 0.0               # Seitenwind-Geschwindigkeit u_cross [m/s]
    headwind: float = 0.0                # Gegenwind [m/s]
    
    # Diskretisierung / Gitter
    nx: int = 240                        # Gitterpunkte in x-Richtung (Breite)
    ny: int = 160                        # Gitterpunkte in y-Richtung (Höhe)
    nz: int = 40                         # Gitterpunkte in z-Richtung (für 3D)
    domain_width_m: float = 120.0        # Domänenbreite [m]
    domain_height_m: float = 80.0        # Domänenhöhe [m]
    domain_length_m: float = 60.0        # Domänenlänge für 3D [m]
    
    # LBM Solver Parameter
    tau: float = 0.55                    # Relaxationszeit tau (tau > 0.5 für Stabilität)
    max_lattice_velocity: float = 0.10   # Maximale Gittergeschwindigkeit u_LB (Inkompressibilitätslimit Ma < 0.15)
    reynolds_number: float = 1000.0      # Skalierte numerische Reynolds-Zahl
    use_numba: bool = True               # Numba JIT-Beschleunigung aktivieren
    parallel_threads: int = 4            # CPU-Threads für Numba / Multi-Core
    
    # Zeitsteuerung
    total_steps: int = 1000              # Gesamtzahl der Zeitschritte
    save_interval: int = 20              # Intervall zum Speichern/Visualisieren
    dt_physical: float = 0.05            # Physikalischer Zeitschritt [s]
    
    def get_aircraft(self) -> AircraftParameters:
        """Gibt das aktuelle Flugzeugobjekt zurück."""
        if self.aircraft_name in AIRCRAFT_PRESETS:
            return AIRCRAFT_PRESETS[self.aircraft_name]
        return self.custom_aircraft
