"""Generate every chart + the demo GIF from results/summary.json (measured data only).

Run: python3 scripts/make_charts.py
Outputs: docs/charts/*.png + docs/demo.gif
Style: dark GPU-theme charts (onavablack background, green bandwidth bars).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "docs" / "charts"
CH.mkdir(parents=True, exist_ok=True)

BG, FG, GREEN, BLUE, ORANGE, GRAY = "#0b1020", "#e8ecf4", "#34d399", "#60a5fa", "#fbbf24", "#64748b"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": GRAY, "grid.color": "#1e293b"})


def load():
    return json.loads((ROOT / "results" / "summary.json").read_text())


def chart_bandwidth(s):
    rows = [r for r in s["vector_add"]["rows"] if r["torch_gbps"]]
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(ns, [r["torch_gbps"] for r in rows], "o-", color=GREEN, lw=2.5, label="PyTorch (1 kernel)")
    tri = [r["variants"][0]["gbps"] if r["variants"] else float("nan") for r in rows]
    plt.plot(ns, tri, "s--", color=BLUE, lw=2, label="Triton BLOCK=256")
    plt.xscale("log")
    plt.xlabel("Vector length N"); plt.ylabel("Bandwidth (GB/s)")
    plt.title("Vector-add: one line of Python, one kernel, ~837 GB/s"); plt.grid(True, alpha=.4)
    plt.legend(framealpha=.15); plt.tight_layout()
    plt.savefig(CH / "bandwidth_vs_size.png", dpi=110); plt.close()


def chart_fusion(s):
    rows = [r for r in s["fusion"]["rows"] if r["speedup"]]
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.axhline(20 / 12, color=GRAY, ls=":", lw=2, label="Traffic theory ceiling 20/12 = 1.667x")
    plt.plot(ns, [r["speedup"] for r in rows], "o-", color=ORANGE, lw=2.5, label="Measured fused/unfused")
    plt.xscale("log"); plt.xlabel("Vector length N"); plt.ylabel("Speedup")
    plt.title("Fusion: measured 1.677x vs 1.667x theory @16.7M"); plt.grid(True, alpha=.4)
    plt.legend(framealpha=.15); plt.tight_layout()
    plt.savefig(CH / "fusion_speedup.png", dpi=110); plt.close()


def chart_attention(s):
    rows = s["attention"]["rows"]
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.loglog(ns, [r["std_GiB"] for r in rows], "o-", color=GREEN, lw=2.5, label="Standard (N^2)")
    plt.loglog(ns, [r["flash_KiB"] / 2**20 for r in rows], "s--", color=BLUE, lw=2,
               label="FlashAttention (flat tile)")
    plt.xlabel("Tokens N"); plt.ylabel("Extra memory (GiB)")
    plt.title("Attention memory: quadratic vs flat (64 GiB vs 2 MiB @32k)")
    plt.grid(True, which="both", alpha=.4); plt.legend(framealpha=.15); plt.tight_layout()
    plt.savefig(CH / "attention_scaling.png", dpi=110); plt.close()


def chart_roofline(s):
    r = s["roofline"]
    plt.figure(figsize=(8, 4.5))
    import math
    xs = [10**(e / 10) for e in range(-20, 41)]
    peak, bw = r["peak_tflops"], r["measured_bw_GBs"] / 1e3
    ys = [min(peak, bw * x) for x in xs]
    plt.loglog(xs, ys, color=GREEN, lw=2.5, label=f"RTX 3090 roof (BW {r['measured_bw_GBs']:.0f} GB/s)")
    plt.loglog([r["vector_add_ai"]], [r["vector_add_attainable_tflops"]], "o", color=ORANGE,
               ms=10, label=f"vector-add: AI=1/12, {r['vector_add_attainable_tflops']:.3f} TFLOP/s")
    plt.xlabel("Arithmetic intensity (FLOP/byte)"); plt.ylabel("Attainable (TFLOP/s)")
    plt.title(f"Roofline: memory-bound (ridge {r['ridge_flop_per_byte']:.1f} FLOP/B)")
    plt.grid(True, which="both", alpha=.4); plt.legend(framealpha=.15); plt.tight_layout()
    plt.savefig(CH / "roofline.png", dpi=110); plt.close()


def demo_gif(s):
    rows = [r for r in s["vector_add"]["rows"] if r["torch_gbps"]]
    labels = [f"N={r['n']:,}" for r in rows]
    torch_v = [r["torch_gbps"] for r in rows]
    tri_v = [r["variants"][0]["gbps"] if r["variants"] else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    def frame(i):
        ax.clear()
        ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
        k = i + 1
        x = range(k)
        ax.bar([p - .2 for p in x], torch_v[:k], .4, color=GREEN, label="PyTorch")
        ax.bar([p + .2 for p in x], tri_v[:k], .4, color=BLUE, label="Triton")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels[:k], rotation=15, fontsize=8)
        ax.set_ylabel("GB/s", color=FG); ax.set_title("Vector-add bandwidth race (measured)", color=FG)
        ax.legend(framealpha=.2)
    FuncAnimation(fig, frame, frames=len(rows), interval=900).save(
        ROOT / "docs" / "demo.gif", writer="pillow", dpi=80)
    plt.close()


def main():
    s = load()
    chart_bandwidth(s); chart_fusion(s); chart_attention(s); chart_roofline(s); demo_gif(s)
    print("charts:", sorted(p.name for p in CH.iterdir()), "+ docs/demo.gif")


if __name__ == "__main__":
    main()
