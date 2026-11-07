#!/usr/bin/env python3
"""Wrapper that regenerates figures without touching original experiment directories."""

from plotting_scripts.plot_figures import (
    figure1_schematic,
    figure2_from_verified_csv,
    figure3_fixed_js,
    figure4_nst,
    figure5_limits,
)

if __name__ == "__main__":
    figure1_schematic()
    figure2_from_verified_csv()
    figure3_fixed_js()
    figure4_nst()
    figure5_limits()
    print("figures regenerated")
