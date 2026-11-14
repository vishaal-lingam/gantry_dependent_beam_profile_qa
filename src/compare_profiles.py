"""
compare_profiles.py

Comparing the Octavius 1500 profile with a water tank scan.

Author:  Vishaal Lingam
Created: 2026-11-14 20:36
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    path = Path(path)
    df = pd.read_excel(path) if path.suffix in (".xlsx", ".xls") else pd.read_csv(path)
    df = df.sort_values("position_mm")
    return df["position_mm"].to_numpy(), df["relative_dose"].to_numpy()


def value_at(x, y, x0):
    return float(np.interp(x0, x, y))


def edges_50(x, y):
    i0 = np.argmin(np.abs(x))
    left = x[: i0 + 1][::-1]
    yl = y[: i0 + 1][::-1]
    k = np.where(yl < 0.5)[0][0]
    xl = np.interp(0.5, [yl[k], yl[k - 1]], [left[k], left[k - 1]])
    right = x[i0:]
    yr = y[i0:]
    k = np.where(yr < 0.5)[0][0]
    xr = np.interp(0.5, [yr[k], yr[k - 1]], [right[k], right[k - 1]])
    return xl, xr


def norm_cax(x, y):
    return y / value_at(x, y, 0.0)


def flat_sym(x, y):
    y = norm_cax(x, y)
    xl, xr = edges_50(x, y)
    W, c = xr - xl, (xr + xl) / 2
    half = 0.4 * W
    d = np.arange(0, half + 1e-9, 0.1)
    left = np.interp(c - d, x, y)
    right = np.interp(c + d, x, y)
    region = np.interp(np.arange(c - half, c + half + 1e-9, 0.1), x, y)
    flat = 100 * (region.max() - region.min()) / (region.max() + region.min())
    ratios = 100 * left / right
    sym = ratios[np.argmax(np.abs(ratios - 100))]
    return dict(width=W, centre=c, flatness=flat, symmetry=sym)


def ratio_stats(xt, yt, xa, ya, region, how):
    lo, hi = region
    tank_at = np.interp(xa, xt, yt)
    m = (xa >= lo) & (xa <= hi)
    if how == "cax":
        a = ya / value_at(xa, ya, 0.0)
        t = tank_at / value_at(xt, yt, 0.0)
    elif how == "mean":
        a = ya / ya[m].mean()
        t = tank_at / tank_at[m].mean()
    else:
        raise ValueError(how)
    r = a[m] / t[m]
    return dict(
        x=xa[m],
        ratio=r,
        spread=100 * (r.max() - r.min()),
        mean=100 * (r.mean() - 1),
        sd=100 * r.std(ddof=1),
        maxdev=100 * np.max(np.abs(r - 1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tank")
    ap.add_argument("octavius")
    ap.add_argument("--label", default="")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    xt, yt = load(args.tank)
    xa, ya = load(args.octavius)

    fs_t = flat_sym(xt, yt)
    fs_a = flat_sym(xa, ya)
    # Comparison region: central 80 % of the TANK field width (reference)
    region = (
        fs_t["centre"] - 0.4 * fs_t["width"],
        fs_t["centre"] + 0.4 * fs_t["width"],
    )
    res = {h: ratio_stats(xt, yt, xa, ya, region, h) for h in ("cax", "mean")}

    print(f"\nProfile comparison {args.label}")
    print(
        f"Region: {region[0]:.1f} to {region[1]:.1f} mm "
        f"({res['cax']['x'].size} chambers, central 80 % of tank field width)\n"
    )
    print("Octavius / tank ratio          CAX-normalised   Mean-normalised")
    for key, lab in [
        ("spread", "Spread, max - min (%)"),
        ("maxdev", "Max |deviation| (%)"),
        ("mean", "Mean deviation (%)"),
        ("sd", "SD of ratios (%)"),
    ]:
        print(f"  {lab:<28}{res['cax'][key]:>10.2f}{res['mean'][key]:>17.2f}")
    print(
        f"  {'Quoted as (Simon style)':<28}{'±%.2f' % (res['cax']['spread'] / 2):>10}"
        f"{'±%.2f' % (res['mean']['spread'] / 2):>17}"
    )

    print("\nBeam parameters                Tank      Octavius   Difference")
    for key, lab, u in [
        ("width", "Field width (mm)", ""),
        ("centre", "Centre (mm)", ""),
        ("flatness", "Flatness (%)", ""),
        ("symmetry", "Symmetry (%)", ""),
    ]:
        print(
            f"  {lab:<28}{fs_t[key]:>8.2f}{fs_a[key]:>12.2f}{fs_a[key] - fs_t[key]:>12.2f}"
        )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(
            2,
            1,
            figsize=(7, 6.5),
            sharex=True,
            gridspec_kw=dict(height_ratios=[1.3, 1]),
        )
        a1.plot(xt, norm_cax(xt, yt), color="0.35", lw=1.2, label="Water tank (CC13)")
        a1.plot(xa, norm_cax(xa, ya), "o", ms=4, color="#1f6fb2", label="Octavius 1500")
        for e in region:
            a1.axvline(e, color="0.7", ls="--", lw=0.8)
            a2.axvline(e, color="0.7", ls="--", lw=0.8)
        a1.set_ylabel("Relative dose (CAX = 1)")
        a1.legend(frameon=False, loc="lower center")
        a1.set_title(
            f"Profile comparison {args.label}".strip(), loc="left", fontsize=11
        )
        for h, col, lab in [
            ("cax", "#c0392b", "CAX normalised"),
            ("mean", "#1f6fb2", "Mean normalised"),
        ]:
            a2.plot(
                res[h]["x"],
                100 * (res[h]["ratio"] - 1),
                "o-",
                ms=3.5,
                lw=0.8,
                color=col,
                label=f"{lab}: spread {res[h]['spread']:.2f} %",
            )
        a2.axhline(0, color="0.3", lw=0.8)
        a2.set_ylabel("Octavius / tank − 1 (%)")
        a2.set_xlabel("Position (mm)")
        a2.legend(frameon=False, fontsize=9)
        for ax in (a1, a2):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
