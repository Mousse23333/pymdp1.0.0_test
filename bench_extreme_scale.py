#!/usr/bin/env python3
"""
Extreme Scale Benchmark
=======================
Push the GB10 to its limits. Test configurations that would be
completely infeasible with legacy NumPy — this is the "emergence at scale" test.

Tests:
1. Massive batch sizes (10k, 50k, 100k agents)
2. Very large state spaces (256, 512, 1024 states per factor)
3. Deep policy horizons (policy_len=1,2,3,4)
4. Combined: large states + large batch
5. Multi-factor at scale with batch

Produces fig9_extreme_scale.png and fig10_frontier.png
"""

import json
import time
import gc
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random as jr
from pymdp.agent import Agent as JAXAgent
from pymdp.legacy import utils as legacy_utils

FIG_DIR = Path("/workspace/AIF_pymdp/figures")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("/workspace/AIF_pymdp/results")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.grid": True, "grid.alpha": 0.3,
})
C_JAX_GPU = "#2ecc71"
C_ACCENT = "#f39c12"
C_DEEP = "#8e44ad"
C_FAIL = "#e74c3c"


def build_and_run(num_obs, num_states, num_controls, batch_size, policy_len=1, n_steps=15, warmup=5):
    """Build JAX agent, run n_steps, return steady-state avg ms and per-agent ms."""
    device = jax.devices("gpu")[0]

    A_np = legacy_utils.random_A_matrix(num_obs, num_states)
    B_np = legacy_utils.random_B_matrix(num_states, num_controls)

    A = [jax.device_put(jnp.broadcast_to(jnp.array(a), (batch_size, *a.shape)), device) for a in A_np]
    B = [jax.device_put(jnp.broadcast_to(jnp.array(b), (batch_size, *b.shape)), device) for b in B_np]
    C = [jax.device_put(jnp.broadcast_to(jnp.zeros(o).at[0].set(1.0), (batch_size, o)), device) for o in num_obs]
    D = [jax.device_put(jnp.broadcast_to(jnp.ones(s) / s, (batch_size, s)), device) for s in num_states]

    agent = JAXAgent(A=A, B=B, C=C, D=D, policy_len=policy_len, batch_size=batch_size)
    rng_key = jr.PRNGKey(0)
    obs = [jnp.zeros((batch_size,), dtype=jnp.int32) for _ in num_obs]
    prior = D

    times = []
    for step in range(n_steps):
        rng_key, subkey = jr.split(rng_key)
        t0 = time.perf_counter()
        qs = agent.infer_states(obs, empirical_prior=prior)
        q_pi, neg_efe = agent.infer_policies(qs)
        keys = jr.split(subkey, batch_size)
        action = agent.sample_action(q_pi, rng_key=keys)
        jax.block_until_ready(action)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        obs = [jr.randint(jr.fold_in(subkey, m), (batch_size,), 0, o) for m, o in enumerate(num_obs)]

    steady = times[warmup:]
    wall_ms = np.mean(steady) * 1000
    per_agent_ms = wall_ms / batch_size
    throughput = batch_size / (wall_ms / 1000)  # agents/sec

    del agent, A, B, C, D, obs, prior
    gc.collect()

    return {
        "wall_ms": float(wall_ms),
        "per_agent_ms": float(per_agent_ms),
        "throughput": float(throughput),
        "first_step_ms": float(times[0] * 1000),
        "all_times_ms": [t * 1000 for t in times],
    }


def safe_run(label, **kwargs):
    """Run benchmark with error handling."""
    print(f"  {label}...", end=" ", flush=True)
    try:
        res = build_and_run(**kwargs)
        print(f"OK  wall={res['wall_ms']:.1f}ms  throughput={res['throughput']:,.0f} agents/s")
        return res
    except Exception as e:
        print(f"FAILED: {e}")
        return {"error": str(e), "wall_ms": 0, "per_agent_ms": 0, "throughput": 0}


def main():
    print("=" * 65)
    print("EXTREME SCALE BENCHMARK — NVIDIA GB10")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 65)

    # Warmup
    x = jnp.ones((100, 100))
    _ = jnp.dot(x, x).block_until_ready()

    results = {}

    # ── Test 1: Massive batch sizes ──────────────────────────────────────
    print("\n[1/5] Massive batch sizes (8 states, 4 controls)")
    batch_sizes = [100, 1000, 5000, 10000, 50000, 100000]
    results["massive_batch"] = {}
    for bs in batch_sizes:
        r = safe_run(f"batch={bs}", num_obs=[8], num_states=[8], num_controls=[4], batch_size=bs)
        results["massive_batch"][f"b={bs}"] = r

    # ── Test 2: Very large state spaces ──────────────────────────────────
    print("\n[2/5] Large state spaces (batch=1)")
    state_sizes = [64, 128, 256, 512, 1024]
    results["large_states"] = {}
    for s in state_sizes:
        nc = min(s, 32)  # cap controls to avoid combinatorial policy explosion
        r = safe_run(f"states={s}", num_obs=[s], num_states=[s], num_controls=[nc], batch_size=1)
        results["large_states"][f"s={s}"] = r

    # ── Test 3: Deep policy horizons ─────────────────────────────────────
    print("\n[3/5] Policy horizon depth (16 states, 4 controls, batch=1)")
    results["policy_depth"] = {}
    for pl in [1, 2, 3, 4]:
        r = safe_run(f"policy_len={pl}", num_obs=[16], num_states=[16], num_controls=[4],
                     batch_size=1, policy_len=pl)
        results["policy_depth"][f"pl={pl}"] = r

    # ── Test 4: Combined large states + large batch ──────────────────────
    print("\n[4/5] Combined scaling (states x batch)")
    combined_configs = [
        (32, 1000), (32, 10000), (32, 50000),
        (64, 1000), (64, 10000), (64, 50000),
        (128, 500), (128, 5000), (128, 10000),
        (256, 100), (256, 1000), (256, 5000),
    ]
    results["combined"] = {}
    for s, bs in combined_configs:
        nc = min(s, 16)
        r = safe_run(f"s={s},b={bs}", num_obs=[s], num_states=[s], num_controls=[nc], batch_size=bs)
        results["combined"][f"s={s}_b={bs}"] = r

    # ── Test 5: Multi-factor at scale with batch ─────────────────────────
    print("\n[5/5] Multi-factor at scale")
    mf_configs = [
        {"label": "2f_b=1000", "obs": [32, 16], "states": [32, 16], "ctrl": [8, 4], "bs": 1000},
        {"label": "2f_b=10000", "obs": [32, 16], "states": [32, 16], "ctrl": [8, 4], "bs": 10000},
        {"label": "3f_b=1000", "obs": [16, 8, 6], "states": [16, 8, 6], "ctrl": [8, 4, 3], "bs": 1000},
        {"label": "3f_b=5000", "obs": [16, 8, 6], "states": [16, 8, 6], "ctrl": [8, 4, 3], "bs": 5000},
        {"label": "4f_b=500", "obs": [16, 8, 6, 4], "states": [16, 8, 6, 4], "ctrl": [8, 4, 3, 2], "bs": 500},
        {"label": "4f_b=1000", "obs": [16, 8, 6, 4], "states": [16, 8, 6, 4], "ctrl": [8, 4, 3, 2], "bs": 1000},
    ]
    results["multi_factor_scale"] = {}
    for cfg in mf_configs:
        r = safe_run(cfg["label"], num_obs=cfg["obs"], num_states=cfg["states"],
                     num_controls=cfg["ctrl"], batch_size=cfg["bs"])
        results["multi_factor_scale"][cfg["label"]] = r

    # ── Save results ─────────────────────────────────────────────────────
    with open(RESULTS_DIR / "extreme_scale.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/extreme_scale.json")

    # ═════════════════════════════════════════════════════════════════════
    # Figure 9: Extreme scale overview (4-panel)
    # ═════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Extreme Scale: Pushing GB10 to the Limits", fontsize=15, fontweight="bold")

    # Panel A: Massive batch throughput
    ax = axes[0, 0]
    mb = results["massive_batch"]
    bs_vals = [int(k.split("=")[1]) for k in mb.keys()]
    tp_vals = [mb[k]["throughput"] for k in mb.keys()]
    ok_mask = [mb[k]["throughput"] > 0 for k in mb.keys()]
    bs_ok = [b for b, m in zip(bs_vals, ok_mask) if m]
    tp_ok = [t for t, m in zip(tp_vals, ok_mask) if m]
    ax.semilogx(bs_ok, [t / 1000 for t in tp_ok], "s-", color=C_JAX_GPU, lw=2, ms=7)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (k agents/s)")
    ax.set_title("A. Throughput vs Batch Size (8 states)")
    for b, t in zip(bs_ok, tp_ok):
        ax.annotate(f"{t/1000:.0f}k", xy=(b, t/1000), xytext=(0, 8),
                   textcoords="offset points", fontsize=8, ha="center", fontweight="bold")

    # Panel B: Large state spaces
    ax = axes[0, 1]
    ls = results["large_states"]
    s_vals = [int(k.split("=")[1]) for k in ls.keys()]
    wall_vals = [ls[k]["wall_ms"] for k in ls.keys()]
    ok_mask = [ls[k]["wall_ms"] > 0 for k in ls.keys()]
    s_ok = [s for s, m in zip(s_vals, ok_mask) if m]
    w_ok = [w for w, m in zip(wall_vals, ok_mask) if m]
    fail_s = [s for s, m in zip(s_vals, ok_mask) if not m]
    ax.plot(s_ok, w_ok, "s-", color=C_JAX_GPU, lw=2, ms=7)
    for fs in fail_s:
        ax.axvline(fs, ls="--", color=C_FAIL, alpha=0.5)
        ax.annotate("OOM/fail", xy=(fs, max(w_ok) if w_ok else 100), fontsize=8, color=C_FAIL, ha="center")
    ax.set_xlabel("State dimension")
    ax.set_ylabel("Wall time per step (ms)")
    ax.set_title("B. Latency vs State-Space Size (b=1)")

    # Panel C: Policy depth
    ax = axes[1, 0]
    pd = results["policy_depth"]
    pl_vals = [int(k.split("=")[1]) for k in pd.keys()]
    w_vals = [pd[k]["wall_ms"] for k in pd.keys()]
    n_pol = []
    for pl in pl_vals:
        n_pol.append(4 ** pl)  # 4 controls ^ policy_len
    ok_mask = [pd[k]["wall_ms"] > 0 for k in pd.keys()]
    pl_ok = [p for p, m in zip(pl_vals, ok_mask) if m]
    w_ok2 = [w for w, m in zip(w_vals, ok_mask) if m]
    np_ok = [n for n, m in zip(n_pol, ok_mask) if m]
    ax.bar(range(len(pl_ok)), w_ok2, color=[C_JAX_GPU if w < 5000 else C_ACCENT for w in w_ok2],
           edgecolor="white", lw=0.5, zorder=3)
    ax.set_xticks(range(len(pl_ok)))
    ax.set_xticklabels([f"len={p}\n({n} policies)" for p, n in zip(pl_ok, np_ok)])
    ax.set_ylabel("Wall time per step (ms)")
    ax.set_title("C. Policy Horizon Depth (16 states, 4 controls)")

    # Panel D: Combined frontier heatmap
    ax = axes[1, 1]
    comb = results["combined"]
    # Build matrix
    unique_s = sorted(set(int(k.split("_")[0].split("=")[1]) for k in comb.keys()))
    unique_b = sorted(set(int(k.split("_")[1].split("=")[1]) for k in comb.keys()))

    tp_matrix = np.zeros((len(unique_s), len(unique_b)))
    for i, s in enumerate(unique_s):
        for j, b in enumerate(unique_b):
            key = f"s={s}_b={b}"
            if key in comb and comb[key]["throughput"] > 0:
                tp_matrix[i, j] = comb[key]["throughput"] / 1000  # k agents/s
            else:
                tp_matrix[i, j] = np.nan

    im = ax.imshow(tp_matrix, aspect="auto", cmap="YlGn", origin="lower",
                   extent=[-0.5, len(unique_b)-0.5, -0.5, len(unique_s)-0.5])
    ax.set_xticks(range(len(unique_b)))
    ax.set_xticklabels([str(b) for b in unique_b], fontsize=8)
    ax.set_yticks(range(len(unique_s)))
    ax.set_yticklabels([str(s) for s in unique_s])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("State dimension")
    ax.set_title("D. Throughput Frontier (k agents/s)")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("k agents/s")

    # Annotate cells
    for i in range(len(unique_s)):
        for j in range(len(unique_b)):
            val = tp_matrix[i, j]
            if not np.isnan(val) and val > 0:
                color = "white" if val > np.nanmax(tp_matrix) * 0.6 else "black"
                ax.text(j, i, f"{val:.0f}k", ha="center", va="center", fontsize=7.5,
                       fontweight="bold", color=color)
            elif np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="grey")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig9_extreme_scale.png")
    plt.close(fig)
    print("  fig9_extreme_scale.png")

    # ═════════════════════════════════════════════════════════════════════
    # Figure 10: Multi-factor at scale
    # ═════════════════════════════════════════════════════════════════════

    fig, ax = plt.subplots(figsize=(10, 5.5))

    mfs = results["multi_factor_scale"]
    labels = list(mfs.keys())
    tp_vals = [mfs[l]["throughput"] / 1000 if mfs[l]["throughput"] > 0 else 0 for l in labels]
    wall_vals = [mfs[l]["wall_ms"] for l in labels]
    colors = [C_JAX_GPU if mfs[l].get("throughput", 0) > 0 else C_FAIL for l in labels]

    x = np.arange(len(labels))
    bars = ax.bar(x, tp_vals, 0.6, color=colors, edgecolor="white", lw=0.5, zorder=3)
    for xi, tp, wt in zip(x, tp_vals, wall_vals):
        if tp > 0:
            ax.text(xi, tp + max(tp_vals) * 0.02, f"{tp:.0f}k/s\n({wt:.0f}ms)",
                   ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (k agent-steps / second)")
    ax.set_title("Multi-Factor Models at Scale on GB10")
    ax.set_ylim(0, max(tp_vals) * 1.25 if tp_vals else 1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_multifactor_scale.png")
    plt.close(fig)
    print("  fig10_multifactor_scale.png")

    # ── Print grand summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXTREME SCALE SUMMARY")
    print("=" * 65)

    max_throughput = 0
    max_config = ""
    for section in results.values():
        for k, v in section.items():
            if isinstance(v, dict) and v.get("throughput", 0) > max_throughput:
                max_throughput = v["throughput"]
                max_config = k

    print(f"\nPeak throughput: {max_throughput:,.0f} agents/s ({max_config})")

    # Find largest successful state space
    largest_state = 0
    for k, v in results.get("large_states", {}).items():
        if v.get("wall_ms", 0) > 0:
            s = int(k.split("=")[1])
            if s > largest_state:
                largest_state = s
    print(f"Largest successful state space: {largest_state} (single agent)")

    # Find largest successful batch
    largest_batch = 0
    for k, v in results.get("massive_batch", {}).items():
        if v.get("wall_ms", 0) > 0:
            b = int(k.split("=")[1])
            if b > largest_batch:
                largest_batch = b
    print(f"Largest successful batch: {largest_batch:,}")


if __name__ == "__main__":
    main()
