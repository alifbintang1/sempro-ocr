"""Generate figures untuk Bab 4 thesis dari hasil comparison."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COMP = json.loads((ROOT / "runs/FinancialStatement-2025-Tahunan-BBNI/comparison.json").read_text(encoding="utf-8"))
OUT_DIR = ROOT / "runs/FinancialStatement-2025-Tahunan-BBNI/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Extract per-approach OVERALL macro rows
overall = [r for r in COMP["rows"] if r["statement"] == "OVERALL (macro)"]

approaches = [r["approach"] for r in overall]
node_f1 = [r["node_f1"] for r in overall]
cell_f1 = [r["cell_f1"] for r in overall]
runtime = [r.get("runtime_s") or 1 for r in overall]
hierarchy = [r["hierarchy_acc"] for r in overall]

# Short labels for display
SHORT = {
    "native_pdf": "native",
    "pymupdf_native": "pymupdf",
    "ocr_full": "ocr",
    "baseline_regex": "regex",
    "pdfplumber_tables": "pdfp-tab",
    "camelot_stream": "camelot",
    "vlm_openai_gpt_4o_mini": "VLM-mini",
    "vlm_openai_gpt_4o": "VLM-gpt4o",
}

# Color by family
def color(a):
    if "vlm" in a: return "#d62728"
    if "ocr" in a: return "#ff7f0e"
    if a in ("native_pdf", "pymupdf_native"): return "#2ca02c"
    return "#1f77b4"

# ── Figure 4.1: Cell F1 vs Node F1 (accuracy trade-off) ─────────────
fig, ax = plt.subplots(figsize=(8, 6))
for a, n, c in zip(approaches, node_f1, cell_f1):
    ax.scatter(n, c, s=140, color=color(a), edgecolor="black", linewidth=0.8, zorder=3)
    dx, dy = 0.012, 0.012
    if SHORT[a] in ("native", "pymupdf"):  # overlapping
        dy = -0.04 if SHORT[a] == "pymupdf" else 0.025
    ax.annotate(SHORT[a], (n, c), xytext=(n + dx, c + dy), fontsize=10)
ax.set_xlabel("Node F1 (structural recovery)", fontsize=12)
ax.set_ylabel("Cell F1 (numeric accuracy)", fontsize=12)
ax.set_title("Gambar 4.1: Trade-off Node F1 vs Cell F1 antar pendekatan", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.0)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_4_1_nodef1_vs_cellf1.png", dpi=150)
print(f"Wrote: {OUT_DIR / 'fig_4_1_nodef1_vs_cellf1.png'}")
plt.close()

# ── Figure 4.2: Runtime vs Node F1 (speed-vs-accuracy) ──────────────
fig, ax = plt.subplots(figsize=(8, 6))
for a, n, t in zip(approaches, node_f1, runtime):
    ax.scatter(t, n, s=140, color=color(a), edgecolor="black", linewidth=0.8, zorder=3)
    ax.annotate(SHORT[a], (t, n), xytext=(t * 1.15, n + 0.012), fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("Runtime (seconds, log scale)", fontsize=12)
ax.set_ylabel("Node F1", fontsize=12)
ax.set_title("Gambar 4.2: Runtime vs Node F1 antar pendekatan", fontsize=12)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(-0.05, 1.0)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_4_2_runtime_vs_nodef1.png", dpi=150)
print(f"Wrote: {OUT_DIR / 'fig_4_2_runtime_vs_nodef1.png'}")
plt.close()

# ── Figure 4.3: Bar chart — metrics per approach (top 4) ────────────
import numpy as np
fig, ax = plt.subplots(figsize=(10, 6))
top4 = sorted(zip(approaches, node_f1, cell_f1, hierarchy),
              key=lambda x: x[1], reverse=True)[:4]
labels_short = [SHORT[a] for a, _, _, _ in top4]
x = np.arange(len(labels_short))
w = 0.27
ax.bar(x - w, [t[1] for t in top4], w, label="Node F1", color="#1f77b4")
ax.bar(x,     [t[2] for t in top4], w, label="Cell F1", color="#2ca02c")
ax.bar(x + w, [t[3] for t in top4], w, label="Hierarchy", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=11)
ax.set_ylabel("Score (0–1)", fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title("Gambar 4.3: Perbandingan tiga metrik utama untuk empat pendekatan teratas",
             fontsize=12)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
for i, (_, n, c, h) in enumerate(top4):
    ax.text(i - w, n + 0.02, f"{n:.2f}", ha="center", fontsize=9)
    ax.text(i,     c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
    ax.text(i + w, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_4_3_top4_metrics.png", dpi=150)
print(f"Wrote: {OUT_DIR / 'fig_4_3_top4_metrics.png'}")
plt.close()

print(f"\nAll figures saved to: {OUT_DIR}")
