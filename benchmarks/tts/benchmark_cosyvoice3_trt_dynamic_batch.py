# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A/B benchmark for CosyVoice3 dynamic-batch TensorRT flow estimation.

The baseline runs B requests sequentially through the legacy CFG=2 TensorRT
profile. The candidate runs the same B requests in one CFM solve through the
dual-profile dynamic engine (CFG batch 2B). This isolates the estimator-side
benefit unlocked by cross-request flow batching without changing the solver.
"""

import argparse
import json
import random
import statistics
import time

import torch
from omegaconf import DictConfig

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
)
from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import (
    build_flow_estimator_trt,
)


def make_cfm(estimator):
    return CausalConditionalCFM(
        in_channels=80,
        cfm_params=DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        ),
        n_spks=1,
        spk_emb_dim=80,
        estimator=estimator,
    )


def make_case(request_batch: int, length: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = 0.1 * torch.randn((request_batch, 80, length), device="cuda", generator=gen)
    mu = 0.1 * torch.randn((request_batch, 80, length), device="cuda", generator=gen)
    mask = torch.ones((request_batch, 1, length), device="cuda")
    spks = 0.1 * torch.randn((request_batch, 80), device="cuda", generator=gen)
    cond = 0.1 * torch.randn((request_batch, 80, length), device="cuda", generator=gen)
    t_span = torch.linspace(0, 1, 11, device="cuda")
    return x, t_span, mu, mask, spks, cond


def slice_case(case, index: int):
    x, t_span, mu, mask, spks, cond = case
    sl = slice(index, index + 1)
    return x[sl], t_span, mu[sl], mask[sl], spks[sl], cond[sl]


def sequential_solve(cfm, case):
    request_batch = int(case[0].shape[0])
    return torch.cat([cfm.solve_euler(*slice_case(case, i)) for i in range(request_batch)], dim=0)


def measure(fn):
    torch.accelerator.synchronize()
    start = time.perf_counter_ns()
    out = fn()
    torch.accelerator.synchronize()
    return out, (time.perf_counter_ns() - start) / 1e6


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_median_ci(values, samples=5000, seed=12345):
    rng = random.Random(seed)
    n = len(values)
    medians = [statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    medians.sort()
    return [
        medians[int(0.025 * samples)],
        medians[min(samples - 1, int(0.975 * samples))],
    ]


def engine_memory_bytes(wrapper):
    engine = wrapper.trt_engine
    value = getattr(engine, "device_memory_size_v2", None)
    if value is None:
        value = getattr(engine, "device_memory_size", None)
    return int(value) if value is not None else None


def run_case(static_cfm, dynamic_cfm, request_batch, length, warmup, repeats):
    case = make_case(
        request_batch=request_batch,
        length=length,
        seed=20260918 + 1000 * request_batch + length,
    )

    def baseline_fn():
        return sequential_solve(static_cfm, case)

    def candidate_fn():
        return dynamic_cfm.solve_euler(*case)

    for _ in range(warmup):
        baseline_fn()
        candidate_fn()
    torch.accelerator.synchronize()

    baseline_ref = baseline_fn()
    candidate_ref = candidate_fn()
    torch.accelerator.synchronize()

    baseline_ms = []
    candidate_ms = []
    delta_ms = []
    speedups = []
    for index in range(repeats):
        if index % 2 == 0:
            _, baseline = measure(baseline_fn)
            _, candidate = measure(candidate_fn)
        else:
            _, candidate = measure(candidate_fn)
            _, baseline = measure(baseline_fn)
        baseline_ms.append(baseline)
        candidate_ms.append(candidate)
        delta_ms.append(baseline - candidate)
        speedups.append(baseline / candidate)

    diff = (baseline_ref - candidate_ref).float()
    denom = float(torch.linalg.vector_norm(baseline_ref.float()).item())
    rel_l2 = float(torch.linalg.vector_norm(diff).item()) / max(denom, 1e-12)

    return {
        "request_batch": request_batch,
        "cfg_batch": 2 * request_batch,
        "length": length,
        "steps": 10,
        "warmup": warmup,
        "repeats": repeats,
        "baseline_sequential_static_ms": summarize(baseline_ms),
        "candidate_dynamic_ms": summarize(candidate_ms),
        "paired_delta_ms": summarize(delta_ms),
        "paired_delta_median_95ci": bootstrap_median_ci(delta_ms),
        "paired_speedup": summarize(speedups),
        "paired_speedup_median_95ci": bootstrap_median_ci(speedups),
        "positive_rounds": sum(value > 0 for value in delta_ms),
        "baseline_nonfinite": int((~torch.isfinite(baseline_ref)).sum().item()),
        "candidate_nonfinite": int((~torch.isfinite(candidate_ref)).sum().item()),
        "relative_l2": rel_l2,
        "max_abs_error": float(diff.abs().max().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", help="CosyVoice3 fp16 flow-estimator ONNX path")
    parser.add_argument("--lengths", type=int, nargs="+", default=[41, 191])
    parser.add_argument("--request-batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-request-batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TensorRT benchmark")
    if max(args.request_batches) > args.max_request_batch:
        raise ValueError("--max-request-batch must cover every requested batch")

    static_wrapper = build_flow_estimator_trt(args.onnx, device="cuda")
    dynamic_wrapper = build_flow_estimator_trt(
        args.onnx,
        device="cuda",
        dynamic_batch=True,
        max_request_batch=args.max_request_batch,
    )
    static_cfm = make_cfm(static_wrapper)
    dynamic_cfm = make_cfm(dynamic_wrapper)

    cases = []
    for length in args.lengths:
        for request_batch in args.request_batches:
            cases.append(
                run_case(
                    static_cfm,
                    dynamic_cfm,
                    request_batch=request_batch,
                    length=length,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "static_io_dtype": str(static_wrapper.io_dtype),
        "dynamic_io_dtype": str(dynamic_wrapper.io_dtype),
        "static_engine_device_memory_bytes": engine_memory_bytes(static_wrapper),
        "dynamic_engine_device_memory_bytes": engine_memory_bytes(dynamic_wrapper),
        "max_request_batch": args.max_request_batch,
        "cases": cases,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
