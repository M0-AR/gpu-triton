"""Run every experiment, print a summary, save results/summary.json."""
from __future__ import annotations
import json
from pathlib import Path
from . import bench_vector_add, bench_fusion, bench_roofline, bench_waves, bench_attention, bench_market

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main():
    import json as _j
    cfg = _j.loads((ROOT / "configs" / "experiments.json").read_text())
    sizes, blocks = cfg["sizes"], cfg["block_sizes"]
    vec = bench_vector_add.run(sizes, block_sizes=blocks)
    measured = [r for r in vec["rows"] if r["torch_gbps"]]
    big = max(measured, key=lambda r: r["n"]) if measured else {"n": 0, "torch_gbps": 1.0}
    roof = bench_roofline.run(big["torch_gbps"] if big["torch_gbps"] else 1.0)
    summary = {
        "vector_add": vec,
        "fusion": bench_fusion.run(sizes),
        "roofline": roof,
        "waves": bench_waves.run(),
        "attention": bench_attention.run(cfg["attention_lengths"], cfg["attention_heads"]),
        "market": bench_market.run(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"device={vec['device']}")
    for r in vec["rows"]:
        v = r["variants"][0] if r["variants"] else None
        print(f"n={r['n']:>9} torch={r['torch_ms']:.3f}ms/{r['torch_gbps']:.1f}GB/s" +
              (f" triton[{v['block']}]={v['ms']:.3f}ms/{v['gbps']:.1f}GB/s err={v['max_err']}" if v else " (cpu-only)"))
    print(f"ridge={roof['ridge_flop_per_byte']:.1f} FLOP/B bound={roof['bound']}")
    print(f"market err={summary['market']['triton']} stats={summary['market']['stats']}")
    print("saved results/summary.json")


if __name__ == "__main__":
    main()
