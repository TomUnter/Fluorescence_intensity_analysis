#!/usr/bin/env python3
"""
Relative fluorescence intensity analysis (whole image) for multichannel stainings.

Expected input: one single-channel image per channel and sample, named
    <sample>_<channel>.<ext>
e.g.   ctrl_1_DAPI.tif, ctrl_1_Phalloidin.tif, ctrl_1_Col1.tif, ctrl_1_Col3.tif
                irr_1_DAPI.tif,  irr_1_Phalloidin.tif,  irr_1_Col1.tif,  irr_1_Col3.tif

The condition is taken from the sample name (the text before the last "_<number>"),
e.g. "ctrl_1" -> condition "ctrl", "irr_2" -> condition "irr".

Usage:
    python intensity_analysis.py path/to/images --control ctrl
    python intensity_analysis.py path/to/images --control ctrl --out results

Output (in --out folder):
    intensities.csv   one row per sample and channel
    summary.csv       mean per condition, relative to control (control = 1.0)
    overview.png      bar plot of relative intensities

IMPORTANT: relative comparisons are only valid if all images were acquired with
identical settings (exposure, gain, laser/LED power, objective).
"""

import argparse
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, filters

# ----------------------------------------------------------------------------
# Settings (adapt names if your files are called differently)
# ----------------------------------------------------------------------------
CHANNELS = {
    "DAPI": "nuclei",
    "Phalloidin": "cells",
    "Col1": "target",
    "Col3": "target",
}
EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
MIN_NUCLEUS_PX = 50  # objects smaller than this are not counted as nuclei


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_gray(path: Path) -> np.ndarray:
    """Load an image as 2D float array. Colour images are converted to grey values."""
    img = io.imread(str(path))
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):  # RGB(A)
            img = img[..., :3].astype(float).mean(axis=-1)
        else:  # e.g. (channels, y, x) -> use the first plane
            img = img[0]
    return img.astype(float)


def background(img: np.ndarray) -> float:
    """Background estimate: median of the darkest 10 % of pixels."""
    cutoff = np.percentile(img, 10)
    return float(np.median(img[img <= cutoff]))


def count_nuclei(dapi: np.ndarray) -> int:
    """Approximate nucleus count (Otsu threshold + connected components)."""
    smooth = filters.gaussian(dapi, sigma=1, preserve_range=True)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = morphology.remove_small_objects(mask, MIN_NUCLEUS_PX)
    return int(measure.label(mask).max())


def cell_area_fraction(phalloidin: np.ndarray) -> float:
    """Fraction of the image covered by phalloidin signal (Otsu threshold)."""
    smooth = filters.gaussian(phalloidin, sigma=1, preserve_range=True)
    return float((smooth > filters.threshold_otsu(smooth)).mean())


def parse_sample(filename: str):
    stem = Path(filename).stem
    for ch in CHANNELS:
        if stem.endswith("_" + ch):
            return stem[: -(len(ch) + 1)], ch
    return None, None


def condition_of(sample: str) -> str:
    return re.sub(r"[_\-]?\d+$", "", sample)


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def analyse(folder: Path) -> pd.DataFrame:
    files = {}
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in EXTENSIONS:
            continue
        sample, ch = parse_sample(f.name)
        if sample:
            files.setdefault(sample, {})[ch] = f

    rows = []
    for sample, chans in files.items():
        missing = set(CHANNELS) - set(chans)
        if missing:
            print(f"[warning] {sample}: missing channels {sorted(missing)}")

        n_nuclei = (
            count_nuclei(load_gray(chans["DAPI"])) if "DAPI" in chans else np.nan
        )
        cell_frac = (
            cell_area_fraction(load_gray(chans["Phalloidin"]))
            if "Phalloidin" in chans
            else np.nan
        )

        for ch, role in CHANNELS.items():
            if ch not in chans:
                continue
            img = load_gray(chans[ch])
            bg = background(img)
            corrected = np.clip(img - bg, 0, None)
            row = {
                "sample": sample,
                "condition": condition_of(sample),
                "channel": ch,
                "mean_raw": img.mean(),
                "background": bg,
                "mean_bg_corrected": corrected.mean(),
                "integrated_bg_corrected": corrected.sum(),
                "n_nuclei_approx": n_nuclei,
                "cell_area_fraction": cell_frac,
            }
            if role == "target":
                row["per_nucleus"] = (
                    corrected.sum() / n_nuclei if n_nuclei else np.nan
                )
                row["per_cell_area"] = (
                    corrected.sum() / (cell_frac * img.size) if cell_frac else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, control: str) -> pd.DataFrame:
    metrics = [c for c in ["mean_bg_corrected", "per_nucleus", "per_cell_area"] if c in df]
    out = []
    for ch, sub in df[df.channel.isin(["Col1", "Col3"])].groupby("channel"):
        grp = sub.groupby("condition")[metrics].agg(["mean", "std", "count"])
        if control not in grp.index:
            print(f"[warning] control condition '{control}' not found for {ch}")
            continue
        for cond in grp.index:
            row = {"channel": ch, "condition": cond}
            for m in metrics:
                ctrl_mean = grp.loc[control, (m, "mean")]
                row[f"{m}_mean"] = grp.loc[cond, (m, "mean")]
                row[f"{m}_std"] = grp.loc[cond, (m, "std")]
                row[f"{m}_n"] = grp.loc[cond, (m, "count")]
                row[f"{m}_relative"] = (
                    grp.loc[cond, (m, "mean")] / ctrl_mean if ctrl_mean else np.nan
                )
            out.append(row)
    return pd.DataFrame(out)


def plot(summary: pd.DataFrame, out: Path):
    channels = summary.channel.unique()
    fig, axes = plt.subplots(1, len(channels), figsize=(4.5 * len(channels), 4), squeeze=False)
    for ax, ch in zip(axes[0], channels):
        sub = summary[summary.channel == ch]
        ax.bar(sub.condition, sub["mean_bg_corrected_relative"])
        ax.axhline(1, color="grey", lw=0.8, ls="--")
        ax.set_title(ch)
        ax.set_ylabel("relative intensity (control = 1)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("folder", type=Path, help="folder with the single-channel images")
    p.add_argument("--control", required=True, help="name of the control condition, e.g. ctrl")
    p.add_argument("--out", type=Path, default=Path("results"), help="output folder")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = analyse(args.folder)
    if df.empty:
        raise SystemExit("No matching images found. Check the file names (<sample>_<channel>.tif).")

    df.to_csv(args.out / "intensities.csv", index=False)
    summary = summarise(df, args.control)
    summary.to_csv(args.out / "summary.csv", index=False)
    if not summary.empty:
        plot(summary, args.out / "overview.png")
    print(df.round(2).to_string(index=False))
    print(f"\nResults written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
