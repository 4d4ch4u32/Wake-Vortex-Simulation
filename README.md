# 🛩️ CFD-Wirbelschleppen-Simulator (Wake Vortex Simulation)

Eine modulare, hochperformante Python-Software zur **2D/3D-CFD-Simulation der Wirbelschleppen von Flugzeugen während des Landeanflugs** unter Berücksichtigung von **Bodeneffekt (Ground Effect)**, **Seitenwind** und **aerodynamischer Zirkulation**.

Die Diskretisierung der inkompressiblen Navier-Stokes-Gleichungen erfolgt mittels der **Lattice-Boltzmann-Methode (LBM)** mit BGK-Kollisionsoperator, erweitert durch Numba-JIT-Beschleunigung für Standard-PCs (8–16 GB RAM, 4+ CPU-Kerne).

---

## 📋 Inhaltsverzeichnis

- [Physikalische & Mathematische Grundlagen](#physikalische--mathematische-grundlagen)
- [Architektur & Modulübersicht](#architektur--modulübersicht)
- [Systemvoraussetzungen & Installation](#systemvoraussetzungen--installation)
- [Schnellstart & Ausführung](#schnellstart--ausführung)
  - [Grafische Benutzeroberfläche (GUI)](#grafische-benutzeroberfläche-gui)
  - [Kommandozeile (CLI)](#kommandozeile-cli)
  - [Analytische Validierung & Benchmarks](#analytische-validierung--benchmarks)
- [Flugzeug-Presets](#flugzeug-presets)
- [Testsuite](#testsuite)

---

## 🔬 Physikalische & Mathematische Grundlagen

### 1. Wirbelerzeugung durch Auftrieb (Kutta-Joukowski & Prandtl)
Ein Flugzeug der Masse $M$ mit Spannweite $b$ und Fluggeschwindigkeit $U_\infty$ erzeugt im stationären Landeanflug ($L = M \cdot g$) eine Gesamtzirkulation:
$$\Gamma_0 = \frac{L}{\rho_\infty U_\infty b_0} = \frac{4 M g}{\pi \rho_\infty U_\infty b}$$
wobei $b_0 = \frac{\pi}{4} b \approx 0{,}7854 \cdot b$ der effektive Wirbelabstand für eine elliptische Auftriebsverteilung ist.

### 2. Absinkgeschwindigkeit & Bodeneffekt
Das gegenläufige Wirbelpaar sinkt im freien Raum mit der gegenseitigen Induktionsgeschwindigkeit ab:
$$w_0 = \frac{\Gamma_0}{2\pi b_0}$$
In Bodennähe ($h < 1{,}5 b_0$) induziert die feste Wand (modelliert durch No-Slip Bounce-Back bzw. Spiegelladungen) ein Auseinanderdriften der Wirbel mit der Geschwindigkeit:
$$v_{\text{drift}}(h) = \frac{\Gamma_0}{2\pi b_0} \cdot \frac{2 (h / b_0)}{1 + 4 (h / b_0)^2} \quad \xrightarrow{h \ll b_0} \quad \frac{\Gamma_0}{4\pi h}$$
Die viskose Wechselwirkung mit der Landebahn erzeugt sekundäre Grenzschichtwirbel entgegengesetzten Drehsinns, die zum charakteristischen **Vortex Rebound (Wiederaufsteigen der Wirbel)** führen.

### 3. Lattice-Boltzmann-Methode (LBM)
- **D2Q9- & D3Q19-Gitter**: Diskretisierung des Geschwindigkeitsraums mit Schallgeschwindigkeit $c_s = 1/\sqrt{3}$.
- **BGK-Kollisionsschritt**:
  $$f_i^*(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} \left(f_i(\mathbf{x}, t) - f_i^{(\text{eq})}(\mathbf{x}, t)\right)$$
  mit $\nu = c_s^2 (\tau - 0{,}5) \Delta t$.
- **Gleichgewichtsverteilung**:
  $$f_i^{(\text{eq})} = w_i \rho \left(1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2}\right)$$
- **Randbedingungen**: Zou-He Geschwindigkeits-Einströmung, konvektiver Ausstrom, Standard No-Slip Bounce-Back am Boden und an Tragflächenprofilen.

---

## 🏗️ Architektur & Modulübersicht

Das Projekt ist strikt modular in voneinander entkoppelte Komponenten unterteilt:

```
landungssimulator/
├── main.py                     # Haupteinstiegspunkt (CLI & GUI Starter)
├── requirements.txt            # Python-Abhängigkeiten
├── README.md                   # Projektdokumentation
├── wake_sim/                   # Hauptpaket
│   ├── config.py               # Flugzeug-Presets (A320, B737, B777, C172) & Parameter
│   ├── preprocessing/          # Vorverarbeitung
│   │   ├── airfoil.py          # NACA 4-Ziffern Profilgenerator & Rasterisierung
│   │   ├── lattice_units.py    # SI <-> LBM Einheitenumrechnung & Re-Skalierung
│   │   └── grid.py             # 2D/3D Rechengitter & Randmasken
│   ├── solver/                 # CFD-Solver
│   │   ├── lbm_d2q9.py         # 2D LBM Solver für NACA-Profilumströmung
│   │   ├── cross_plane_wake.py # 2D Cross-Plane Landungs- & Bodeneffekt-Solver
│   │   ├── lbm_d3q19.py        # 3D D3Q19 LBM Solver für Randwirbel
│   │   ├── numba_kernels.py    # JIT-beschleunigte Rechenkerne (mit NumPy Fallback)
│   │   └── boundary_conditions.py # Zou-He, Bounce-Back, Freistromgrenzen
│   ├── postprocessing/         # Analyse & Auswertung
│   │   ├── aerodynamics.py     # Zirkulation Gamma(r), Enstrophie, Tangentialprofil
│   │   ├── vortex_tracker.py   # Wirbelkerndetektion, Trajektorie & Rebound-Erkennung
│   │   └── metrics.py          # ICAO-Staffelungsabstände & Energieerhaltung
│   ├── validation/             # Validierungsmodelle
│   │   ├── prandtl_lifting_line.py # Prandtl Traglinientheorie (Fourier-Reihe)
│   │   ├── vortex_models.py    # Lamb-Oseen, Burnham-Hallock, Spiegelladungen
│   │   └── benchmark.py        # Automatisierter Vergleich & Fehlermetriken (RMSE, R^2)
│   ├── visualization/          # Visualisierung
│   │   ├── live_plotter.py     # Matplotlib 4-Panel Echtzeit-Dashboard & Reports
│   │   └── export.py           # PNG/PDF-Grafikexport, GIF-Animation & CSV-Export
│   └── ui/                     # Benutzeroberflächen
│       ├── cli.py              # Vollständiges CLI mit Argparse
│       └── gui_tk.py           # Intuitive Tkinter Desktop-GUI
└── tests/                      # Pytest Testsuite (26 Unit- & Integrationstests)
```

---

## 💻 Systemvoraussetzungen & Installation

### Voraussetzungen:
- Python 3.8 oder neuer
- Betriebssystem: Linux, macOS oder Windows
- Empfohlene Hardware: Standard-PC mit 8–16 GB RAM und Mehrkern-CPU

### Installation:

```bash
# Repository klonen oder Verzeichnis öffnen
cd /home/anton/projects/landungssimulator

# Abhängigkeiten installieren
pip install -r requirements.txt
```

Inhalt von `requirements.txt`:
- `numpy>=1.22.0`
- `scipy>=1.8.0`
- `matplotlib>=3.5.0`
- `numba>=0.56.0` (für optionale JIT-Beschleunigung)
- `pytest>=7.0.0`

---

## 🚀 Schnellstart & Ausführung

### 🖥️ Grafische Benutzeroberfläche (GUI)
Startet die minimalistische Desktop-GUI mit Schiebereglern, Live-Plot-Dashboard und Ein-Klick-Export:

```bash
python3 main.py --gui
```

### ⌨️ Kommandozeile (CLI)

#### 1. Standard-Landesimulation (Airbus A320)
Simuliert das Absinken des Wirbelpaars im Querschnitt und exportiert Grafik und Telemetriedaten:
```bash
python3 main.py --mode cross_plane --aircraft A320 --steps 300 --save-plot output/a320_wake.png --save-csv output/a320_data.csv
```

#### 2. Simulation mit Seitenwind und Echtzeit-Animation
```bash
python3 main.py --mode cross_plane --aircraft B777 --crosswind 3.0 --height 50.0 --visualize
```

#### 3. 2D NACA 0012 Profilumströmung
Berechnet Auftriebs- ($C_L$) und Widerstandsbeiwerte ($C_D$) sowie Wirbelablösung:
```bash
python3 main.py --mode airfoil --profile 0012 --aoa 6.0 --speed 70.0 --steps 500
```

#### 4. 3D-Flügelumströmung mit Randwirbeln (D3Q19)
```bash
python3 main.py --mode 3d --aircraft A320 --steps 100
```

#### 5. Erstellung einer animierten GIF-Datei
```bash
python3 main.py --mode cross_plane --aircraft A320 --steps 300 --save-gif output/landing_wake.gif
```

---

## 📊 Analytische Validierung & Benchmarks

Das Validierungsmodul vergleicht die CFD-Ergebnisse quantitativ mit etablierten analytischen Theorien:
- **Zirkulation**: Vergleich mit Prandtl's Traglinientheorie ($\Gamma_0$).
- **Wirbelprofil**: Berechnung von $R^2$ und RMSE gegenüber dem **Burnham-Hallock-** und **Lamb-Oseen-Modell**.
- **Trajektorie**: Vergleich des Absinkens und Ausweichens im Bodeneffekt gegen die **Spiegelladungsmethode**.

Ausführung über die CLI:
```bash
python3 main.py --mode validate --aircraft A320 --steps 100 --save-plot output/validation_report.png
```

---

## ✈️ Flugzeug-Presets

| Preset | Flugzeugmuster | MTOW [kg] | Spannweite $b$ [m] | Landegeschwindigkeit $U_\infty$ [m/s] | ICAO-Kategorie |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `A320` | Airbus A320-200 | 66.000 | 35,8 | 70,0 (~136 kts) | Medium |
| `B737` | Boeing 737-800 | 65.000 | 35,8 | 72,0 (~140 kts) | Medium |
| `B777` | Boeing 777-300ER | 250.000 | 64,8 | 75,0 (~146 kts) | Heavy |
| `C172` | Cessna 172 Skyhawk | 1.100 | 11,0 | 30,0 (~58 kts) | Light |

---

## 🧪 Testsuite

Das Projekt enthält 26 automatisierte Pytest-Tests, die alle Komponenten (Geometrie, Gitter, LBM-Solver, Numba-Kerne, Postprocessing, Validierungsmodelle, GUI/CLI) abdecken.

Ausführung der Tests:
```bash
pytest -v
```

## Verwendeter Prompt

```
Erstelle eine Python-basierte Software zur 2D/3D-Simulation der Wirbelschleppen eines Flugzeugs während der Landung mithilfe numerischer Strömungsmechanik (CFD). Nutze die Lattice-Boltzmann-Methode oder Finite-Volumen-Methode für die Diskretisierung der Navier-Stokes-Gleichungen. Die Simulation soll:
Physikalische Parameter wie Luftdichte, Viskosität, Flugzeuggeometrie (z. B. Flügelprofil NACA 0012) und Anströmgeschwindigkeit (z. B. 50–100 m/s) berücksichtigen.
Wirbelbildung (z. B. durch Auftriebskräfte) und Zeitentwicklung der Schleppen modellieren.
Echtzeit-Visualisierung der Strömungsfelder (Geschwindigkeit, Druck, Wirbelstärke) via Matplotlib oder PyVista ermöglichen.
Optimiert für normale Hardware (CPU/GPU-Beschleunigung optional via Numba/CUDA) sein.
Modular aufgebaut sein (z. B. Separation von Preprocessing, Solver, Postprocessing).
Einfache Benutzeroberfläche (CLI oder minimalistische GUI mit Tkinter/PyQt) zur Parameter-Eingabe bieten.
Validierung durch Vergleich mit analytischen Lösungen (z. B. Prandtl’s Lifting-Line-Theorie) oder experimentellen Daten ermöglichen.
Gib den vollständigen Code inkl. Kommentare, Anforderungen (requirements.txt), und eine kurze Anleitung zur Ausführung. Nutze Bibliotheken wie NumPy, SciPy, OpenFOAM-Python (falls verfügbar), oder FEniCS für die numerische Implementierung. Priorisiere Einfachheit und Performance für einen Standard-PC (8–16 GB RAM, 4+ CPU-Kerne).
```

LLM Model: Gemini 3.7 Flash
