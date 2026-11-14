"""
syn_data.py

Generate synthetic data for validating Octavius 1500 profile against water tank scan.

Author:  Vishaal Lingam
Created: 2026-11-14 20:05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FIELD = 200.0  # nominal field width at measurement point (mm)
SIGMA_PEN = 3.5  # penumbra sigma (mm)
HORN = 0.025  # horn height relative to centre
TILT = 0.004  # linear asymetry across field


def true_profile(x):
    h = FIELD / 2
    u = np.clip(x / h, -1.2, 1.2)
    core = 1 + HORN * (1.6 * u**2 - 0.6 * u**4) - TILT * u / 2
    edges = 0.5 * (
        erf((h - x) / (np.sqrt(2) * SIGMA_PEN))
        - erf((-h - x) / (np.sqrt(2) * SIGMA_PEN))
    )
    tail = 0.02 * np.exp(-np.abs(x) / 60)
    return core * edges + tail * (1 - edges)


def detector_response(x, width):
    offs = np.linspace(-width / 2, width / 2, 21)
    return np.mean([true_profile(x + o) for o in offs], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--center-error",
        type=float,
        default=0.6,
        help="calibration error of the central chamber, %% "
        "(Single-point normalisation bias)",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Water tank: CC13 ~ 6 mm active diameter, 0.5 mm steps, 0.1 % noise
    xt = np.arange(-150, 150.01, 0.5)
    tank = detector_response(xt, 6.0) * (1 + rng.normal(0, 0.001, xt.size))
    tank /= tank[np.argmin(np.abs(xt))]

    # Octavius 1500: 4.4 mm chambers, merged rows -> 5 mm spacing,
    # 27 x 27 cm array -> positions -135..135 mm.
    xa = np.arange(-135, 135.01, 5.0)
    calib_err = rng.normal(0, 0.003, xa.size)  # 0.3 % chamber-to-chamber
    calib_err[np.argmin(np.abs(xa))] = args.center_error / 100
    arr = (
        detector_response(xa, 4.4)
        * (1 + calib_err)
        * (1 + rng.normal(0, 0.002, xa.size))
    )
    arr /= arr[np.argmin(np.abs(xa))]

    pd.DataFrame({"position_mm": xt, "relative_dose": tank}).to_excel(
        DATA_DIR / "tank_6X_20x20_crossline.xlsx", index=False
    )
    pd.DataFrame({"position_mm": xa, "relative_dose": arr}).to_excel(
        DATA_DIR / "octavius_6X_20x20_crossline.xlsx", index=False
    )
    print(f"Saved to {DATA_DIR}")
    print(
        f"Wrote tank ({xt.size} points) and Octavius ({xa.size} points) profiles "
        f"(central chamber error {args.center_error:+.2f} %)"
    )


if __name__ == "__main__":
    main()
