#!/usr/bin/env python3
"""
Generate publication-quality benchmark visualisation for pymdp 1.0.0.

Reads the JSON results and produces 6 figures saved to /workspace/AIF_pymdp/figures/
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS = Path("/workspace/AIF_pymdp/results/benchmark_20260409_130517.json")
FIG_DIR = Path("/workspace/AIF_pymdp/figures")
FIG_DIR.mkdir(exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

C_LEGACY = "#e74c3c"
C_JAX_GPU = "#2ecc71"
C_JAX_CPU = "#3498db"
C_ACCENT = "#f39c12"

def total_ms(entry):
    """Sum of mean times across operations, in milliseconds."""
    return sum(v["mean"] for v in entry.values() if isinstance(v, dict) and "mean" in v) * 1000


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: State-space scaling – 3-backend comparison (log scale)
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5.5))

scaling = data["scaling"]
sizes = []
leg_vals, gpu_vals, cpu_vals = [], [], []

for label, entry in scaling.items():
    s = int(label.split("=")[1])
    sizes.append(s)
    leg_vals.append(total_ms(entry["legacy_numpy"]))
    gpu_vals.append(total_ms(entry["jax_gpu"]))
    cpu_vals.append(total_ms(entry["jax_cpu"]))

ax.semilogy(sizes, leg_vals, "o-", color=C_LEGACY, lw=2.2, ms=7, label="Legacy NumPy", zorder=3)
ax.semilogy(sizes, gpu_vals, "s-", color=C_JAX_GPU, lw=2.2, ms=7, label="JAX GPU (GB10)", zorder=3)
ax.semilogy(sizes, cpu_vals, "^-", color=C_JAX_CPU, lw=2.2, ms=7, label="JAX CPU (Grace)", zorder=3)

# Mark crossover
cross_x = 200
ax.axvline(cross_x, ls="--", color="grey", alpha=0.5)
ax.annotate("GPU overtakes\nNumPy here",
            xy=(cross_x, leg_vals[-1]), xytext=(cross_x - 65, leg_vals[-1] * 3),
            fontsize=9, ha="center", color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=1))

ax.set_xlabel("State-space dimension (per factor)")
ax.set_ylabel("Per-step latency (ms, log scale)")
ax.set_title("State-Space Scaling: Legacy NumPy vs JAX Backends")
ax.legend(loc="upper left")
ax.set_xticks(sizes)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_state_scaling.png")
plt.close(fig)
print("  fig1_state_scaling.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Batch scaling – total wall time AND per-agent throughput
# ═════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

batch_data = data["batch_scaling"]
bsizes, wall_ms, per_agent_ms = [], [], []
for label, entry in batch_data.items():
    bs = int(label.split("=")[1])
    bsizes.append(bs)
    wt = total_ms(entry)
    wall_ms.append(wt)
    per_agent_ms.append(wt / bs)

# NumPy baseline per-agent
numpy_per_agent = total_ms(data["small_model"]["legacy_numpy"])

# Left: wall time
ax1.semilogx(bsizes, wall_ms, "s-", color=C_JAX_GPU, lw=2.2, ms=7)
ax1.set_xlabel("Batch size (number of concurrent agents)")
ax1.set_ylabel("Total wall-clock time per step (ms)")
ax1.set_title("JAX GPU: Near-Constant Wall Time")
ax1.annotate(f"1 agent: {wall_ms[0]:.0f} ms\n10k agents: {wall_ms[-1]:.0f} ms\n(only {wall_ms[-1]/wall_ms[0]:.1f}x more)",
             xy=(bsizes[-1], wall_ms[-1]), xytext=(500, wall_ms[0] + 18),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_JAX_GPU, lw=1.2))

# Right: per-agent cost
ax2.loglog(bsizes, per_agent_ms, "s-", color=C_JAX_GPU, lw=2.2, ms=7, label="JAX GPU per-agent")
ax2.axhline(numpy_per_agent, ls="--", color=C_LEGACY, lw=1.8, label=f"NumPy baseline ({numpy_per_agent:.2f} ms)")
ax2.fill_between([min(bsizes), max(bsizes)], 0, numpy_per_agent,
                 alpha=0.08, color=C_JAX_GPU)
ax2.set_xlabel("Batch size (number of concurrent agents)")
ax2.set_ylabel("Per-agent latency (ms, log scale)")
ax2.set_title("Per-Agent Cost Drops with Batch Size")
ax2.legend(loc="upper right")

# Mark parity
for i in range(len(bsizes) - 1):
    if per_agent_ms[i] > numpy_per_agent and per_agent_ms[i+1] <= numpy_per_agent:
        parity_bs = int(np.interp(numpy_per_agent, [per_agent_ms[i+1], per_agent_ms[i]], [bsizes[i+1], bsizes[i]]))
        ax2.axvline(parity_bs, ls=":", color=C_ACCENT, lw=1.5)
        ax2.annotate(f"Parity ~{parity_bs}",
                     xy=(parity_bs, numpy_per_agent), xytext=(parity_bs * 3, numpy_per_agent * 5),
                     fontsize=9, color=C_ACCENT,
                     arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1))
        break

# Add speedup labels
for bs, pa in zip(bsizes[-3:], per_agent_ms[-3:]):
    sp = numpy_per_agent / pa
    ax2.annotate(f"{sp:.0f}x", xy=(bs, pa), xytext=(bs, pa * 0.35),
                 fontsize=9, fontweight="bold", ha="center", color=C_JAX_GPU)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_batch_scaling.png")
plt.close(fig)
print("  fig2_batch_scaling.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Multi-factor comparison – the money chart
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5.5))

mf = data["multi_factor"]
labels_mf = list(mf.keys())
n_policies = [8, 32, 96, 192]
leg_mf = [total_ms(mf[l]["legacy_numpy"]) for l in labels_mf]
gpu_mf = [total_ms(mf[l]["jax_gpu"]) for l in labels_mf]

x = np.arange(len(labels_mf))
w = 0.35

bars_leg = ax.bar(x - w/2, leg_mf, w, color=C_LEGACY, label="Legacy NumPy", zorder=3)
bars_gpu = ax.bar(x + w/2, gpu_mf, w, color=C_JAX_GPU, label="JAX GPU", zorder=3)

ax.set_yscale("log")
ax.set_xlabel("Model complexity")
ax.set_ylabel("Per-step latency (ms, log scale)")
ax.set_title("Multi-Factor Models: Where JAX Wins Decisively")
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n({p} policies)" for l, p in zip(labels_mf, n_policies)])
ax.legend(loc="upper left")

# Add speedup annotations
for i, (lg, gp) in enumerate(zip(leg_mf, gpu_mf)):
    ratio = lg / gp
    if ratio > 1:
        ax.annotate(f"JAX {ratio:.1f}x faster",
                    xy=(i, max(lg, gp)), xytext=(i, max(lg, gp) * 2.5),
                    fontsize=9, fontweight="bold", ha="center", color=C_JAX_GPU,
                    arrowprops=dict(arrowstyle="->", color=C_JAX_GPU, lw=1))
    else:
        ax.annotate(f"NumPy {1/ratio:.0f}x faster",
                    xy=(i, max(lg, gp)), xytext=(i, max(lg, gp) * 2.5),
                    fontsize=9, ha="center", color=C_LEGACY,
                    arrowprops=dict(arrowstyle="->", color=C_LEGACY, lw=1))

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_multi_factor.png")
plt.close(fig)
print("  fig3_multi_factor.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Operation breakdown – stacked bar (small model)
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

sm = data["small_model"]
backends = ["legacy_numpy", "jax_cpu", "jax_gpu"]
backend_labels = ["Legacy NumPy", "JAX CPU", "JAX GPU"]
ops = ["infer_states", "infer_policies", "sample_action"]
op_colors = ["#3498db", "#e74c3c", "#2ecc71"]
op_labels = ["infer_states", "infer_policies", "sample_action"]

x = np.arange(len(backends))
bottom = np.zeros(len(backends))
for op, color, olabel in zip(ops, op_colors, op_labels):
    vals = [sm[b][op]["mean"] * 1000 for b in backends]
    ax.bar(x, vals, 0.5, bottom=bottom, color=color, label=olabel, zorder=3)
    # Label with value
    for xi, v, bot in zip(x, vals, bottom):
        if v > 0.5:  # only label if visible
            ax.text(xi, bot + v/2, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottom += vals

ax.set_ylabel("Per-step latency (ms)")
ax.set_title("Operation Breakdown: Small Model (5 states, 3 controls)")
ax.set_xticks(x)
ax.set_xticklabels(backend_labels)
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_operation_breakdown.png")
plt.close(fig)
print("  fig4_operation_breakdown.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5: GPU stress – throughput heatmap-like chart
# ═════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))

stress = data["gpu_stress"]
configs = list(stress.keys())
states_list = [32, 32, 64, 64, 128, 128]
batch_list = [1000, 5000, 1000, 5000, 500, 1000]

wall_times = []
throughputs = []
for cfg in configs:
    entry = stress[cfg]
    if "error" in entry:
        wall_times.append(0)
        throughputs.append(0)
    else:
        wt = total_ms(entry)
        wall_times.append(wt)
        bs = int(cfg.split("b=")[1])
        throughputs.append(bs / (wt / 1000))  # agents/sec

x = np.arange(len(configs))
bars = ax.bar(x, throughputs, 0.6, color=C_JAX_GPU, zorder=3, edgecolor="white", lw=0.5)

for xi, tp, wt in zip(x, throughputs, wall_times):
    ax.text(xi, tp + 400, f"{tp:,.0f}\nagents/s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.text(xi, tp / 2, f"{wt:.0f} ms", ha="center", va="center", fontsize=8, color="white")

ax.set_ylabel("Throughput (agent-steps / second)")
ax.set_title("GPU Stress Test: Throughput on NVIDIA GB10")
ax.set_xticks(x)
ax.set_xticklabels([c.replace(",", "\n") for c in configs], fontsize=9)
ax.set_ylim(0, max(throughputs) * 1.35)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_gpu_stress.png")
plt.close(fig)
print("  fig5_gpu_stress.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6: Summary dashboard – the "one chart to show them all"
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 10))
fig.suptitle("pymdp 1.0.0 Benchmark Dashboard — NVIDIA GB10 (Blackwell)", fontsize=15, fontweight="bold", y=0.98)

# ── Panel A: State scaling ────────────────────────────────────────────────
ax_a = fig.add_subplot(2, 3, 1)
ax_a.semilogy(sizes, leg_vals, "o-", color=C_LEGACY, lw=2, ms=5, label="NumPy")
ax_a.semilogy(sizes, gpu_vals, "s-", color=C_JAX_GPU, lw=2, ms=5, label="JAX GPU")
ax_a.semilogy(sizes, cpu_vals, "^-", color=C_JAX_CPU, lw=2, ms=5, label="JAX CPU")
ax_a.set_xlabel("State dimension")
ax_a.set_ylabel("Latency (ms)")
ax_a.set_title("A. State-Space Scaling")
ax_a.legend(fontsize=8)

# ── Panel B: Batch scaling ────────────────────────────────────────────────
ax_b = fig.add_subplot(2, 3, 2)
ax_b.loglog(bsizes, per_agent_ms, "s-", color=C_JAX_GPU, lw=2, ms=5, label="JAX GPU per-agent")
ax_b.axhline(numpy_per_agent, ls="--", color=C_LEGACY, lw=1.5, label=f"NumPy ({numpy_per_agent:.2f} ms)")
ax_b.set_xlabel("Batch size")
ax_b.set_ylabel("Per-agent latency (ms)")
ax_b.set_title("B. Batch Scaling (per-agent)")
ax_b.legend(fontsize=8)

# ── Panel C: Multi-factor bars ────────────────────────────────────────────
ax_c = fig.add_subplot(2, 3, 3)
x_c = np.arange(len(labels_mf))
ax_c.bar(x_c - 0.2, leg_mf, 0.35, color=C_LEGACY, label="NumPy")
ax_c.bar(x_c + 0.2, gpu_mf, 0.35, color=C_JAX_GPU, label="JAX GPU")
ax_c.set_yscale("log")
ax_c.set_xticks(x_c)
ax_c.set_xticklabels([f"{p}p" for p in n_policies], fontsize=9)
ax_c.set_xlabel("Policies")
ax_c.set_ylabel("Latency (ms)")
ax_c.set_title("C. Multi-Factor Models")
ax_c.legend(fontsize=8)

# ── Panel D: Speedup ratio ───────────────────────────────────────────────
ax_d = fig.add_subplot(2, 3, 4)
# Compute speedup: >1 means JAX faster, <1 means NumPy faster
speedups_scaling = [l / g for l, g in zip(leg_vals, gpu_vals)]
ax_d.plot(sizes, speedups_scaling, "s-", color=C_JAX_GPU, lw=2, ms=5, label="State scaling")
speedups_mf = [l / g for l, g in zip(leg_mf, gpu_mf)]
ax_d.plot(n_policies, speedups_mf, "D-", color=C_ACCENT, lw=2, ms=5, label="Multi-factor")
ax_d.axhline(1.0, ls="--", color="grey", lw=1)
ax_d.fill_between([0, max(max(sizes), max(n_policies))], 0, 1, alpha=0.05, color=C_LEGACY)
ax_d.fill_between([0, max(max(sizes), max(n_policies))], 1, max(max(speedups_scaling), max(speedups_mf)) * 1.2,
                   alpha=0.05, color=C_JAX_GPU)
ax_d.text(10, 0.5, "NumPy faster", fontsize=8, color=C_LEGACY, alpha=0.7)
ax_d.text(10, max(speedups_mf) * 0.7, "JAX faster", fontsize=8, color=C_JAX_GPU, alpha=0.7)
ax_d.set_xlabel("Dimension / Policies")
ax_d.set_ylabel("Speedup (JAX GPU / NumPy)")
ax_d.set_title("D. Speedup Ratio")
ax_d.legend(fontsize=8)

# ── Panel E: Throughput bar ──────────────────────────────────────────────
ax_e = fig.add_subplot(2, 3, 5)
# Throughput = agents/sec for batch scaling
throughput_batch = [bs / (wt / 1000) for bs, wt in zip(bsizes, wall_ms)]
ax_e.semilogx(bsizes, throughput_batch, "s-", color=C_JAX_GPU, lw=2, ms=5)
ax_e.set_xlabel("Batch size")
ax_e.set_ylabel("Throughput (agents/s)")
ax_e.set_title("E. Agent Throughput (JAX GPU)")
ax_e.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

# ── Panel F: Inference methods ───────────────────────────────────────────
ax_f = fig.add_subplot(2, 3, 6)
inf_data = data["inference_methods"]
methods = list(inf_data.keys())
means = [inf_data[m]["mean"] * 1000 for m in methods]
stds = [inf_data[m]["std"] * 1000 for m in methods]
bars = ax_f.barh(methods, means, xerr=stds, color=[C_JAX_GPU, C_JAX_CPU, C_ACCENT],
                  edgecolor="white", lw=0.5, capsize=3, zorder=3)
ax_f.set_xlabel("Latency (ms)")
ax_f.set_title("F. Inference Methods (32 states)")
for bar, m in zip(bars, means):
    ax_f.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
              f"{m:.2f} ms", va="center", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "fig6_dashboard.png")
plt.close(fig)
print("  fig6_dashboard.png")

print(f"\nAll figures saved to {FIG_DIR}/")
