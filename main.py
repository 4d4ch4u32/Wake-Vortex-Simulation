#!/usr/bin/env python3
"""
CFD-Wirbelschleppen-Simulator (Wake Vortex CFD Simulation)
==========================================================
Haupteinstiegspunkt für CLI- und GUI-Ausführung.
"""

import sys
from wake_sim.ui.cli import run_cli


def main():
    """Haupteinstiegsfunktion."""
    run_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
