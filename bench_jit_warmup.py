#!/usr/bin/env python3
"""
JIT Warmup Analysis
===================
Measure per-step latency for every single step of a long run,
showing that JIT compilation overhead is a one-time cost and
steady-state performance is dramatically different from cold-start.

Produces:
  - fig7_jit_warmup.png       — per-step latency over time (cold → warm)
  - fig8_amortized_cost.png   — cumulative average latency vs num steps
  - jit_warmup_data.json      — raw per-step timing data
"""

import json
import time
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random as jr
from pymdp.agent import Agent as JAXAgent
from pymdp.legacy.agent import Agent as LegacyAgent
from pymdp.legacy import utils as legacy_utils

FIG_DIR = Path("/workspace/AIF_pymdp/figures")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("/workspace/AIF_pymdp/results")

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.grid": True, "grid.alpha": 0.3,
})
C_LEGACY = "#e74c3c"
C_JAX_GPU = "#2ecc71"
C_JAX_CPU = "#3498db"
C_WARM = "#27ae60"
C_COLD = "#e67e22"


def run_jax_steps(num_obs, num_states, num_controls, batch_size, n_steps, device="gpu"):
    """Run JAX agent for n_steps, returning per-step wall time for EACH step."""
    jax_device = jax.devices(device)[0] if device != "cpu" else jax.devices("cpu")[0]

    A_np = legacy_utils.random_A_matrix(num_obs, num_states)
    B_np = legacy_utils.random_B_matrix(num_states, num_controls)

    A = [jax.device_put(jnp.broadcast_to(jnp.array(a), (batch_size, *a.shape)), jax_device) for a in A_np]
    B = [jax.device_put(jnp.broadcast_to(jnp.array(b), (batch_size, *b.shape)), jax_device) for b in B_np]
    C = [jax.device_put(jnp.broadcast_to(jnp.zeros(o), (batch_size, o)).at[:, 0].set(1.0), jax_device)
         for o in num_obs]
    D = [jax.device_put(jnp.broadcast_to(jnp.ones(s) / s, (batch_size, s)), jax_device)
         for s in num_states]

    agent = JAXAgent(A=A, B=B, C=C, D=D, policy_len=1, batch_size=batch_size)
    rng_key = jr.PRNGKey(0)
    obs = [jnp.zeros((batch_size,), dtype=jnp.int32) for _ in num_obs]
    empirical_prior = D

    step_times = []
    for step in range(n_steps):
        rng_key, subkey = jr.split(rng_key)
        t0 = time.perf_counter()
        qs = agent.infer_states(obs, empirical_prior=empirical_prior)
        q_pi, neg_efe = agent.infer_policies(qs)
        keys = jr.split(subkey, batch_size)
        action = agent.sample_action(q_pi, rng_key=keys)
        jax.block_until_ready(action)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        obs = [jr.randint(jr.fold_in(subkey, m), (batch_size,), 0, o) for m, o in enumerate(num_obs)]

    return step_times


def run_legacy_steps(num_obs, num_states, num_controls, n_steps):
    """Run Legacy NumPy agent, returning per-step wall time."""
    A = legacy_utils.random_A_matrix(num_obs, num_states)
    B = legacy_utils.random_B_matrix(num_states, num_controls)
    C = legacy_utils.obj_array_zeros(num_obs)
    C[0][0] = 1.0
    D = legacy_utils.obj_array_uniform(num_states)

    agent = LegacyAgent(A=A, B=B, C=C, D=D, policy_len=1)
    obs = [0] * len(num_obs)
    step_times = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        qs = agent.infer_states(obs)
        q_pi, efe = agent.infer_policies()
        action = agent.sample_action()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        obs = [np.random.randint(0, o) for o in num_obs]
    return step_times


def main():
    N_STEPS = 200

    # ── Config 1: Small model ────────────────────────────────────────────
    print("Running small model (5 states)...")
    num_obs, num_states, num_controls = [5], [5], [3]

    jax_gpu_small = run_jax_steps(num_obs, num_states, num_controls, batch_size=1, n_steps=N_STEPS, device="gpu")
    legacy_small = run_legacy_steps(num_obs, num_states, num_controls, n_steps=N_STEPS)

    # ── Config 2: Medium model with batch ─────────────────────────────────
    print("Running medium model (16 states, batch=1000)...")
    num_obs2, num_states2, num_controls2 = [16], [16], [8]
    jax_gpu_batch = run_jax_steps(num_obs2, num_states2, num_controls2, batch_size=1000, n_steps=N_STEPS, device="gpu")

    # ── Config 3: Multi-factor model ──────────────────────────────────────
    print("Running multi-factor model (3 factors)...")
    num_obs3, num_states3, num_controls3 = [16, 8, 6], [16, 8, 6], [8, 4, 3]
    jax_gpu_mf = run_jax_steps(num_obs3, num_states3, num_controls3, batch_size=1, n_steps=N_STEPS, device="gpu")
    legacy_mf = run_legacy_steps(num_obs3, num_states3, num_controls3, n_steps=N_STEPS)

    # Save raw data
    raw_data = {
        "small_jax_gpu": jax_gpu_small,
        "small_legacy": legacy_small,
        "batch_jax_gpu": jax_gpu_batch,
        "mf_jax_gpu": jax_gpu_mf,
        "mf_legacy": legacy_mf,
    }
    with open(RESULTS_DIR / "jit_warmup_data.json", "w") as f:
        json.dump(raw_data, f, indent=2)

    # ═════════════════════════════════════════════════════════════════════
    # Figure 7: Per-step latency timeline (JIT warmup visible)
    # ═════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    steps = np.arange(1, N_STEPS + 1)

    # Panel A: Small model
    ax = axes[0]
    ax.plot(steps, np.array(jax_gpu_small) * 1000, color=C_JAX_GPU, lw=1, alpha=0.8, label="JAX GPU")
    ax.axhline(np.mean(legacy_small) * 1000, ls="--", color=C_LEGACY, lw=1.5, label=f"NumPy avg ({np.mean(legacy_small)*1000:.2f} ms)")
    # Mark warmup zone
    warmup_end = 5  # usually first few steps
    ax.axvspan(0, warmup_end, alpha=0.1, color=C_COLD)
    ax.annotate("JIT\ncompilation", xy=(2.5, max(jax_gpu_small[:5]) * 1000 * 0.8),
                fontsize=8, ha="center", color=C_COLD, fontweight="bold")
    cold_avg = np.mean(jax_gpu_small[:warmup_end]) * 1000
    warm_avg = np.mean(jax_gpu_small[warmup_end:]) * 1000
    ax.set_title(f"A. Small (5 states, b=1)\nCold: {cold_avg:.0f}ms  Warm: {warm_avg:.1f}ms")
    ax.set_xlabel("Step")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: Batch model
    ax = axes[1]
    ax.plot(steps, np.array(jax_gpu_batch) * 1000, color=C_JAX_GPU, lw=1, alpha=0.8, label="JAX GPU (b=1000)")
    ax.axvspan(0, warmup_end, alpha=0.1, color=C_COLD)
    cold_b = np.mean(jax_gpu_batch[:warmup_end]) * 1000
    warm_b = np.mean(jax_gpu_batch[warmup_end:]) * 1000
    ax.annotate("JIT\ncompilation", xy=(2.5, max(jax_gpu_batch[:5]) * 1000 * 0.8),
                fontsize=8, ha="center", color=C_COLD, fontweight="bold")
    ax.set_title(f"B. Medium (16 states, b=1000)\nCold: {cold_b:.0f}ms  Warm: {warm_b:.1f}ms")
    ax.set_xlabel("Step")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=8, loc="upper right")

    # Panel C: Multi-factor
    ax = axes[2]
    ax.plot(steps, np.array(jax_gpu_mf) * 1000, color=C_JAX_GPU, lw=1, alpha=0.8, label="JAX GPU")
    ax.axhline(np.mean(legacy_mf) * 1000, ls="--", color=C_LEGACY, lw=1.5, label=f"NumPy avg ({np.mean(legacy_mf)*1000:.0f} ms)")
    ax.axvspan(0, warmup_end, alpha=0.1, color=C_COLD)
    cold_mf = np.mean(jax_gpu_mf[:warmup_end]) * 1000
    warm_mf = np.mean(jax_gpu_mf[warmup_end:]) * 1000
    ax.annotate("JIT\ncompilation", xy=(2.5, max(jax_gpu_mf[:5]) * 1000 * 0.8),
                fontsize=8, ha="center", color=C_COLD, fontweight="bold")
    ax.set_title(f"C. 3-Factor (96 policies)\nCold: {cold_mf:.0f}ms  Warm: {warm_mf:.1f}ms")
    ax.set_xlabel("Step")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("JIT Compilation is a One-Time Cost", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_jit_warmup.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig7_jit_warmup.png")

    # ═════════════════════════════════════════════════════════════════════
    # Figure 8: Amortized cost — cumulative average over N steps
    # ═════════════════════════════════════════════════════════════════════

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Small model — cumulative avg
    cum_jax = np.cumsum(jax_gpu_small) / np.arange(1, N_STEPS + 1) * 1000
    cum_leg = np.cumsum(legacy_small) / np.arange(1, N_STEPS + 1) * 1000

    ax1.plot(steps, cum_jax, color=C_JAX_GPU, lw=2, label="JAX GPU (amortized)")
    ax1.plot(steps, cum_leg, color=C_LEGACY, lw=2, label="NumPy (amortized)")
    ax1.set_xlabel("Total steps completed")
    ax1.set_ylabel("Cumulative average latency (ms/step)")
    ax1.set_title("A. Small Model: Amortized Cost Convergence")
    ax1.legend(fontsize=9)
    # Annotate final values
    ax1.annotate(f"JAX: {cum_jax[-1]:.1f} ms", xy=(N_STEPS, cum_jax[-1]),
                xytext=(N_STEPS - 40, cum_jax[-1] + 20), fontsize=9, color=C_JAX_GPU)
    ax1.annotate(f"NumPy: {cum_leg[-1]:.2f} ms", xy=(N_STEPS, cum_leg[-1]),
                xytext=(N_STEPS - 40, cum_leg[-1] + 10), fontsize=9, color=C_LEGACY)

    # Panel B: Multi-factor — cumulative avg
    cum_jax_mf = np.cumsum(jax_gpu_mf) / np.arange(1, N_STEPS + 1) * 1000
    cum_leg_mf = np.cumsum(legacy_mf) / np.arange(1, N_STEPS + 1) * 1000

    ax2.plot(steps, cum_jax_mf, color=C_JAX_GPU, lw=2, label="JAX GPU (amortized)")
    ax2.plot(steps, cum_leg_mf, color=C_LEGACY, lw=2, label="NumPy (amortized)")
    ax2.set_xlabel("Total steps completed")
    ax2.set_ylabel("Cumulative average latency (ms/step)")
    ax2.set_title("B. 3-Factor Model: JAX Amortization Pays Off Quickly")
    ax2.legend(fontsize=9)

    # Find crossover step
    diff = cum_jax_mf - cum_leg_mf
    crossover_candidates = np.where(diff < 0)[0]
    if len(crossover_candidates) > 0:
        cross_step = crossover_candidates[0] + 1
        ax2.axvline(cross_step, ls=":", color=C_WARM, lw=1.5)
        ax2.annotate(f"JAX amortized cost\nbeats NumPy at step {cross_step}",
                     xy=(cross_step, cum_jax_mf[cross_step - 1]),
                     xytext=(cross_step + 30, cum_jax_mf[cross_step - 1] + 200),
                     fontsize=9, color=C_WARM, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_WARM, lw=1.2))

    ax2.annotate(f"JAX: {cum_jax_mf[-1]:.0f} ms", xy=(N_STEPS, cum_jax_mf[-1]),
                xytext=(N_STEPS - 50, cum_jax_mf[-1] - 150), fontsize=9, color=C_JAX_GPU)
    ax2.annotate(f"NumPy: {cum_leg_mf[-1]:.0f} ms", xy=(N_STEPS, cum_leg_mf[-1]),
                xytext=(N_STEPS - 50, cum_leg_mf[-1] + 80), fontsize=9, color=C_LEGACY)

    fig.suptitle("Amortized Cost: JIT Overhead Disappears Over Time", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_amortized_cost.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_amortized_cost.png")

    # ═════════════════════════════════════════════════════════════════════
    # Print summary
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("JIT WARMUP ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nSmall model (5 states, batch=1):")
    print(f"  First step (cold):     {jax_gpu_small[0]*1000:.0f} ms")
    print(f"  Steps 2-5 (warm-up):   {np.mean(jax_gpu_small[1:5])*1000:.1f} ms avg")
    print(f"  Steps 6+ (steady):     {np.mean(jax_gpu_small[5:])*1000:.1f} ms avg")
    print(f"  NumPy steady:          {np.mean(legacy_small)*1000:.2f} ms avg")
    print(f"  Compilation overhead:  {jax_gpu_small[0]*1000 - np.mean(jax_gpu_small[5:])*1000:.0f} ms (one-time)")

    print(f"\nMedium model (16 states, batch=1000):")
    print(f"  First step (cold):     {jax_gpu_batch[0]*1000:.0f} ms")
    print(f"  Steps 6+ (steady):     {np.mean(jax_gpu_batch[5:])*1000:.1f} ms avg")
    print(f"  Per-agent steady:      {np.mean(jax_gpu_batch[5:])*1000/1000:.4f} ms")

    print(f"\n3-Factor model (96 policies, batch=1):")
    print(f"  First step (cold):     {jax_gpu_mf[0]*1000:.0f} ms")
    print(f"  Steps 6+ (steady):     {np.mean(jax_gpu_mf[5:])*1000:.1f} ms avg")
    print(f"  NumPy steady:          {np.mean(legacy_mf)*1000:.0f} ms avg")
    print(f"  Steady-state speedup:  {np.mean(legacy_mf)/np.mean(jax_gpu_mf[5:]):.1f}x (JAX faster)")
    if len(crossover_candidates) > 0:
        print(f"  Amortization crossover: step {cross_step}")


if __name__ == "__main__":
    main()
