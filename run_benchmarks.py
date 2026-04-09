#!/usr/bin/env python3
"""
AIF_pymdp Benchmark Suite
=========================
Comprehensive benchmarks comparing pymdp 1.0.0 JAX backend vs legacy NumPy backend.

Tests cover:
1. State inference (exact, fpi, ovf, mmp, vmp)
2. Policy inference / EFE computation
3. Action sampling
4. Full agent loop (infer_states -> infer_policies -> sample_action)
5. Batch scaling (JAX vmap)
6. State-space scaling
7. MCTS vs exhaustive policy search
8. GPU vs CPU (JAX)

Results are saved to /workspace/AIF_pymdp/results/
"""

import os
import sys
import json
import time
import gc
import warnings
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import numpy as np

# ── JAX setup ────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from jax import random as jr

# ── pymdp imports ────────────────────────────────────────────────────────────
import pymdp
from pymdp.agent import Agent as JAXAgent
from pymdp import utils as jax_utils
from pymdp import inference as jax_inference
from pymdp import control as jax_control

from pymdp.legacy.agent import Agent as LegacyAgent
from pymdp.legacy import utils as legacy_utils
from pymdp.legacy import inference as legacy_inference
from pymdp.legacy import control as legacy_control

RESULTS_DIR = Path("/workspace/AIF_pymdp/results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer():
    """Simple context-manager timer returning elapsed seconds."""
    t = {"elapsed": 0.0}
    start = time.perf_counter()
    yield t
    t["elapsed"] = time.perf_counter() - start


def warmup_jax():
    """Run a throwaway computation so JIT compilation cost is excluded."""
    x = jnp.ones((100, 100))
    _ = jnp.dot(x, x).block_until_ready()


def jax_sync():
    """Block until all JAX computations finish (for accurate timing)."""
    jax.block_until_ready(jnp.zeros(1))


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_legacy_model(num_obs, num_states, num_controls):
    """Build a random generative model for the legacy (NumPy) agent."""
    num_modalities = len(num_obs)
    num_factors = len(num_states)

    A = legacy_utils.random_A_matrix(num_obs, num_states)
    B = legacy_utils.random_B_matrix(num_states, num_controls)

    # Legacy agent expects obj_arrays (numpy structured arrays)
    C = legacy_utils.obj_array_zeros(num_obs)
    C[0][0] = 1.0
    D = legacy_utils.obj_array_uniform(num_states)

    return A, B, C, D


def build_jax_model(num_obs, num_states, num_controls, batch_size=1):
    """Build a random generative model for the JAX agent."""
    num_modalities = len(num_obs)
    num_factors = len(num_states)

    A_np = legacy_utils.random_A_matrix(num_obs, num_states)
    B_np = legacy_utils.random_B_matrix(num_states, num_controls)

    # Convert to JAX arrays and add batch dimension
    A = [jnp.broadcast_to(jnp.array(a), (batch_size, *a.shape)) for a in A_np]
    B = [jnp.broadcast_to(jnp.array(b), (batch_size, *b.shape)) for b in B_np]
    C = [jnp.broadcast_to(jnp.zeros(o), (batch_size, o)) for o in num_obs]
    D = [jnp.broadcast_to(jnp.ones(s) / s, (batch_size, s)) for s in num_states]

    # Set preference
    C[0] = C[0].at[:, 0].set(1.0)

    return A, B, C, D


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1: Legacy NumPy vs JAX – core operations
# ─────────────────────────────────────────────────────────────────────────────

def bench_legacy_agent_loop(num_obs, num_states, num_controls, n_steps=50):
    """Run full agent loop with legacy NumPy backend."""
    A, B, C, D = build_legacy_model(num_obs, num_states, num_controls)
    agent = LegacyAgent(A=A, B=B, C=C, D=D, policy_len=1)

    obs = [0] * len(num_obs)
    timings = {"infer_states": [], "infer_policies": [], "sample_action": []}

    for step in range(n_steps):
        with timer() as t:
            qs = agent.infer_states(obs)
        timings["infer_states"].append(t["elapsed"])

        with timer() as t:
            q_pi, efe = agent.infer_policies()
        timings["infer_policies"].append(t["elapsed"])

        with timer() as t:
            action = agent.sample_action()
        timings["sample_action"].append(t["elapsed"])

        # Generate next observation (random)
        obs = [np.random.randint(0, o) for o in num_obs]

    return {k: {"mean": np.mean(v[5:]), "std": np.std(v[5:]), "min": np.min(v[5:]), "max": np.max(v[5:])}
            for k, v in timings.items()}


def bench_jax_agent_loop(num_obs, num_states, num_controls, batch_size=1, n_steps=50, device="gpu"):
    """Run full agent loop with JAX backend."""
    jax_device = jax.devices(device)[0] if device != "cpu" else jax.devices("cpu")[0]

    A, B, C, D = build_jax_model(num_obs, num_states, num_controls, batch_size)

    # Place arrays on target device
    A = [jax.device_put(a, jax_device) for a in A]
    B = [jax.device_put(b, jax_device) for b in B]
    C = [jax.device_put(c, jax_device) for c in C]
    D = [jax.device_put(d, jax_device) for d in D]

    agent = JAXAgent(A=A, B=B, C=C, D=D, policy_len=1, batch_size=batch_size)
    rng_key = jr.PRNGKey(0)

    obs = [jnp.zeros((batch_size,), dtype=jnp.int32) for _ in num_obs]
    empirical_prior = D

    timings = {"infer_states": [], "infer_policies": [], "sample_action": []}

    for step in range(n_steps):
        rng_key, subkey = jr.split(rng_key)

        with timer() as t:
            qs = agent.infer_states(obs, empirical_prior=empirical_prior)
            jax_sync()
        timings["infer_states"].append(t["elapsed"])

        with timer() as t:
            q_pi, neg_efe = agent.infer_policies(qs)
            jax_sync()
        timings["infer_policies"].append(t["elapsed"])

        with timer() as t:
            keys = jr.split(subkey, batch_size)
            action = agent.sample_action(q_pi, rng_key=keys)
            jax_sync()
        timings["sample_action"].append(t["elapsed"])

        # Random next observation
        obs = [jr.randint(jr.fold_in(subkey, m), (batch_size,), 0, o) for m, o in enumerate(num_obs)]

    return {k: {"mean": float(np.mean(v[5:])), "std": float(np.std(v[5:])),
                "min": float(np.min(v[5:])), "max": float(np.max(v[5:]))}
            for k, v in timings.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2: State-space scaling
# ─────────────────────────────────────────────────────────────────────────────

def bench_scaling(sizes, batch_size=1, n_steps=30):
    """Test how performance scales with state-space dimensionality."""
    results = {}
    for s in sizes:
        label = f"states={s}"
        print(f"  Scaling test: {label}")
        num_obs = [s]
        num_states = [s]
        num_controls = [s]

        legacy_res = bench_legacy_agent_loop(num_obs, num_states, num_controls, n_steps=n_steps)

        jax_gpu_res = bench_jax_agent_loop(num_obs, num_states, num_controls,
                                           batch_size=batch_size, n_steps=n_steps, device="gpu")
        jax_cpu_res = bench_jax_agent_loop(num_obs, num_states, num_controls,
                                           batch_size=batch_size, n_steps=n_steps, device="cpu")

        results[label] = {
            "legacy_numpy": legacy_res,
            "jax_gpu": jax_gpu_res,
            "jax_cpu": jax_cpu_res,
        }
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3: Batch scaling (JAX only)
# ─────────────────────────────────────────────────────────────────────────────

def bench_batch_scaling(batch_sizes, num_obs=(5,), num_states=(5,), num_controls=(3,), n_steps=30):
    """Test how JAX GPU performance scales with batch size."""
    results = {}
    for bs in batch_sizes:
        label = f"batch={bs}"
        print(f"  Batch test: {label}")
        jax_res = bench_jax_agent_loop(
            list(num_obs), list(num_states), list(num_controls),
            batch_size=bs, n_steps=n_steps, device="gpu"
        )
        results[label] = jax_res
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 4: Multi-factor models
# ─────────────────────────────────────────────────────────────────────────────

def bench_multi_factor(n_steps=30):
    """Test performance with increasing number of state factors."""
    configs = [
        {"label": "1-factor", "num_obs": [16], "num_states": [16], "num_controls": [8]},
        {"label": "2-factor", "num_obs": [16, 8], "num_states": [16, 8], "num_controls": [8, 4]},
        {"label": "3-factor", "num_obs": [16, 8, 6], "num_states": [16, 8, 6], "num_controls": [8, 4, 3]},
        {"label": "4-factor", "num_obs": [16, 8, 6, 4], "num_states": [16, 8, 6, 4], "num_controls": [8, 4, 3, 2]},
    ]
    results = {}
    for cfg in configs:
        print(f"  Multi-factor test: {cfg['label']}")
        legacy_res = bench_legacy_agent_loop(cfg["num_obs"], cfg["num_states"], cfg["num_controls"], n_steps)
        jax_res = bench_jax_agent_loop(cfg["num_obs"], cfg["num_states"], cfg["num_controls"],
                                       batch_size=1, n_steps=n_steps, device="gpu")
        results[cfg["label"]] = {"legacy_numpy": legacy_res, "jax_gpu": jax_res}
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 5: Inference methods comparison (JAX)
# ─────────────────────────────────────────────────────────────────────────────

def bench_inference_methods(n_steps=30):
    """Compare different state inference methods in the JAX backend."""
    num_obs = [32]
    num_states = [32]
    num_controls = [8]
    batch_size = 1

    A, B, C, D = build_jax_model(num_obs, num_states, num_controls, batch_size)
    methods = ["exact", "fpi", "ovf"]

    results = {}
    for method in methods:
        print(f"  Inference method: {method}")
        agent = JAXAgent(A=A, B=B, C=C, D=D, policy_len=1, inference_algo=method, batch_size=batch_size)
        rng_key = jr.PRNGKey(0)
        obs = [jnp.zeros((batch_size,), dtype=jnp.int32)]
        empirical_prior = D

        timings = []
        for step in range(n_steps):
            rng_key, subkey = jr.split(rng_key)
            with timer() as t:
                qs = agent.infer_states(obs, empirical_prior=empirical_prior)
                jax_sync()
            timings.append(t["elapsed"])
            obs = [jr.randint(subkey, (batch_size,), 0, num_obs[0])]

        results[method] = {
            "mean": float(np.mean(timings[5:])),
            "std": float(np.std(timings[5:])),
            "min": float(np.min(timings[5:])),
            "max": float(np.max(timings[5:])),
        }
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 6: Large batch + large state-space (deep stress test for GPU)
# ─────────────────────────────────────────────────────────────────────────────

def bench_gpu_stress(n_steps=20):
    """Push GPU with large batches on larger models."""
    configs = [
        {"label": "s=32,b=1000", "num_obs": [32], "num_states": [32], "num_controls": [8], "batch_size": 1000},
        {"label": "s=32,b=5000", "num_obs": [32], "num_states": [32], "num_controls": [8], "batch_size": 5000},
        {"label": "s=64,b=1000", "num_obs": [64], "num_states": [64], "num_controls": [16], "batch_size": 1000},
        {"label": "s=64,b=5000", "num_obs": [64], "num_states": [64], "num_controls": [16], "batch_size": 5000},
        {"label": "s=128,b=500", "num_obs": [128], "num_states": [128], "num_controls": [32], "batch_size": 500},
        {"label": "s=128,b=1000", "num_obs": [128], "num_states": [128], "num_controls": [32], "batch_size": 1000},
    ]
    results = {}
    for cfg in configs:
        label = cfg["label"]
        print(f"  GPU stress: {label}")
        try:
            res = bench_jax_agent_loop(
                cfg["num_obs"], cfg["num_states"], cfg["num_controls"],
                batch_size=cfg["batch_size"], n_steps=n_steps, device="gpu"
            )
            results[label] = res
        except Exception as e:
            print(f"    FAILED: {e}")
            results[label] = {"error": str(e)}
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 7: MCTS vs exhaustive policy search
# ─────────────────────────────────────────────────────────────────────────────

def bench_mcts_vs_exhaustive(n_steps=20):
    """Compare MCTS planning vs exhaustive EFE-based policy evaluation."""
    from pymdp.planning.mcts import mcts_policy_search

    num_obs = [16]
    num_states = [16]
    num_controls = [8]
    batch_size = 1

    A, B, C, D = build_jax_model(num_obs, num_states, num_controls, batch_size)

    results = {}

    # Exhaustive (default)
    print("  Planning: exhaustive EFE")
    agent_exh = JAXAgent(A=A, B=B, C=C, D=D, policy_len=2, batch_size=batch_size)
    rng_key = jr.PRNGKey(42)
    obs = [jnp.zeros((batch_size,), dtype=jnp.int32)]
    empirical_prior = D

    timings_exh = []
    for step in range(n_steps):
        rng_key, subkey = jr.split(rng_key)
        qs = agent_exh.infer_states(obs, empirical_prior=empirical_prior)
        with timer() as t:
            q_pi, neg_efe = agent_exh.infer_policies(qs)
            jax_sync()
        timings_exh.append(t["elapsed"])
        keys = jr.split(subkey, batch_size)
        action = agent_exh.sample_action(q_pi, rng_key=keys)
        obs = [jr.randint(subkey, (batch_size,), 0, num_obs[0])]

    results["exhaustive"] = {
        "mean": float(np.mean(timings_exh[3:])),
        "std": float(np.std(timings_exh[3:])),
        "num_policies": int(agent_exh.policies.num_policies),
    }

    # MCTS
    print("  Planning: MCTS (num_simulations=256)")
    try:
        mcts_fn = mcts_policy_search(max_depth=4, num_simulations=256)
        agent_mcts = JAXAgent(A=A, B=B, C=C, D=D, policy_len=2, batch_size=batch_size, planning_fn=mcts_fn)
        rng_key = jr.PRNGKey(42)
        obs = [jnp.zeros((batch_size,), dtype=jnp.int32)]
        empirical_prior = D

        timings_mcts = []
        for step in range(n_steps):
            rng_key, subkey = jr.split(rng_key)
            qs = agent_mcts.infer_states(obs, empirical_prior=empirical_prior)
            with timer() as t:
                q_pi, info = agent_mcts.infer_policies(qs, rng_key=subkey)
                jax_sync()
            timings_mcts.append(t["elapsed"])
            keys = jr.split(subkey, batch_size)
            action = agent_mcts.sample_action(q_pi, rng_key=keys)
            obs = [jr.randint(subkey, (batch_size,), 0, num_obs[0])]

        results["mcts_256"] = {
            "mean": float(np.mean(timings_mcts[3:])),
            "std": float(np.std(timings_mcts[3:])),
        }
    except Exception as e:
        print(f"    MCTS FAILED: {e}")
        results["mcts_256"] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("AIF_pymdp Benchmark Suite")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 70)

    warmup_jax()
    all_results = {
        "meta": {
            "date": datetime.now().isoformat(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "pymdp_package": "inferactively-pymdp 1.0.0",
        }
    }

    # ── Test 1: Small model comparison (Legacy vs JAX GPU vs JAX CPU) ────
    print("\n[1/7] Legacy NumPy vs JAX – small model (obs=5, states=5, controls=3)")
    num_obs, num_states, num_controls = [5], [5], [3]
    legacy_small = bench_legacy_agent_loop(num_obs, num_states, num_controls, n_steps=100)
    jax_gpu_small = bench_jax_agent_loop(num_obs, num_states, num_controls, batch_size=1, n_steps=100, device="gpu")
    jax_cpu_small = bench_jax_agent_loop(num_obs, num_states, num_controls, batch_size=1, n_steps=100, device="cpu")
    all_results["small_model"] = {
        "config": {"num_obs": num_obs, "num_states": num_states, "num_controls": num_controls},
        "legacy_numpy": legacy_small,
        "jax_gpu": jax_gpu_small,
        "jax_cpu": jax_cpu_small,
    }
    print(f"  Legacy total mean: {sum(v['mean'] for v in legacy_small.values()):.6f}s")
    print(f"  JAX GPU total mean: {sum(v['mean'] for v in jax_gpu_small.values()):.6f}s")
    print(f"  JAX CPU total mean: {sum(v['mean'] for v in jax_cpu_small.values()):.6f}s")

    # ── Test 2: State-space scaling ──────────────────────────────────────
    print("\n[2/7] State-space scaling")
    scaling_sizes = [5, 10, 20, 50, 100, 200]
    all_results["scaling"] = bench_scaling(scaling_sizes, batch_size=1, n_steps=30)

    # ── Test 3: Batch scaling (JAX GPU) ──────────────────────────────────
    print("\n[3/7] Batch scaling (JAX GPU)")
    batch_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    all_results["batch_scaling"] = bench_batch_scaling(batch_sizes, n_steps=30)

    # ── Test 4: Multi-factor models ──────────────────────────────────────
    print("\n[4/7] Multi-factor models")
    all_results["multi_factor"] = bench_multi_factor(n_steps=30)

    # ── Test 5: Inference methods ────────────────────────────────────────
    print("\n[5/7] Inference methods comparison")
    all_results["inference_methods"] = bench_inference_methods(n_steps=30)

    # ── Test 6: GPU stress test ──────────────────────────────────────────
    print("\n[6/7] GPU stress test (large batch + large state-space)")
    all_results["gpu_stress"] = bench_gpu_stress(n_steps=20)

    # ── Test 7: MCTS vs exhaustive ───────────────────────────────────────
    print("\n[7/7] MCTS vs exhaustive policy search")
    all_results["mcts_vs_exhaustive"] = bench_mcts_vs_exhaustive(n_steps=20)

    # ── Save results ─────────────────────────────────────────────────────
    out_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Small model speedup
    leg_total = sum(v["mean"] for v in legacy_small.values())
    gpu_total = sum(v["mean"] for v in jax_gpu_small.values())
    cpu_total = sum(v["mean"] for v in jax_cpu_small.values())
    print(f"\nSmall model (5 states) per-step latency:")
    print(f"  Legacy NumPy : {leg_total*1000:.3f} ms")
    print(f"  JAX CPU      : {cpu_total*1000:.3f} ms")
    print(f"  JAX GPU      : {gpu_total*1000:.3f} ms")
    if gpu_total > 0:
        print(f"  Speedup (Legacy/GPU): {leg_total/gpu_total:.1f}x")
    if cpu_total > 0:
        print(f"  Speedup (Legacy/CPU): {leg_total/cpu_total:.1f}x")

    # Batch scaling summary
    print(f"\nBatch scaling (JAX GPU, 5 states):")
    for label, res in all_results["batch_scaling"].items():
        total = sum(v["mean"] for v in res.values())
        print(f"  {label:>12s}: {total*1000:.3f} ms total per step")

    # Scaling summary
    print(f"\nState-space scaling (per-step total, ms):")
    print(f"  {'Size':>6s}  {'Legacy':>10s}  {'JAX GPU':>10s}  {'JAX CPU':>10s}  {'Speedup':>8s}")
    for label, res in all_results["scaling"].items():
        leg = sum(v["mean"] for v in res["legacy_numpy"].values())
        gpu = sum(v["mean"] for v in res["jax_gpu"].values())
        cpu = sum(v["mean"] for v in res["jax_cpu"].values())
        sp = leg / gpu if gpu > 0 else float("inf")
        print(f"  {label:>6s}  {leg*1000:>10.3f}  {gpu*1000:>10.3f}  {cpu*1000:>10.3f}  {sp:>7.1f}x")

    # GPU stress summary
    print(f"\nGPU stress test (per-step total, ms):")
    for label, res in all_results["gpu_stress"].items():
        if "error" in res:
            print(f"  {label:>20s}: FAILED - {res['error'][:60]}")
        else:
            total = sum(v["mean"] for v in res.values())
            print(f"  {label:>20s}: {total*1000:.3f} ms")

    # MCTS summary
    print(f"\nMCTS vs Exhaustive (policy inference, ms):")
    for label, res in all_results["mcts_vs_exhaustive"].items():
        if "error" in res:
            print(f"  {label:>12s}: FAILED - {res['error'][:60]}")
        else:
            print(f"  {label:>12s}: {res['mean']*1000:.3f} ms (±{res['std']*1000:.3f})")

    return all_results


if __name__ == "__main__":
    results = main()
