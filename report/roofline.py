"""Merge results/*.csv, print a table, and plot achieved bandwidth as a fraction of peak.

    python report/roofline.py                       # reads results/*.csv, writes results/summary.md + png
    python report/roofline.py --peak-gbps 1792      # RTX 5090 spec-sheet DRAM bandwidth (GB/s); verify for your card

Percent-of-peak uses the *ideal* byte count (read inputs once, write outputs once), so a kernel
that re-reads data from cache can legitimately exceed 100 percent; interpret with that in mind.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--peak-gbps", type=float, default=1792.0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.results}/*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {args.results}/ -- run the benchmarks first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["pct_peak_bw"] = 100 * df["gbps"] / args.peak_gbps

    md = ["| kernel | shape | impl | ms | GB/s | % peak BW | TFLOP/s |", "|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["kernel", "shape", "impl"]).iterrows():
        tf = f"{r.tflops:.2f}" if r.tflops > 0 else "-"
        md.append(f"| {r.kernel} | {r['shape']} | {r.impl} | {r.ms:.4f} | {r.gbps:.0f} | {r.pct_peak_bw:.0f} | {tf} |")
    out = Path(args.results)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    kernels = [k for k in df.kernel.unique()]
    fig, axes = plt.subplots(1, len(kernels), figsize=(4.2 * len(kernels), 4), squeeze=False)
    for ax, k in zip(axes[0], kernels):
        sub = df[df.kernel == k]
        piv = sub.pivot_table(index="shape", columns="impl", values="pct_peak_bw")
        piv.plot.bar(ax=ax, rot=30)
        ax.axhline(100, color="k", lw=0.8, ls="--")
        ax.set_title(k); ax.set_ylabel("% of peak DRAM BW (ideal bytes)"); ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "bandwidth.png", dpi=140)
    print(f"wrote {out / 'summary.md'} and {out / 'bandwidth.png'}")


if __name__ == "__main__":
    main()
