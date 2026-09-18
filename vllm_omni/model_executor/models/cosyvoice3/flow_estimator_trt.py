# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TensorRT engine for the CosyVoice3 flow-decoder (CFM) DiT estimator.

The estimator is the per-step network the conditional flow-matching ODE solver
calls during code2wav (token -> mel). It dominates code2wav latency; the
upstream CausalConditionalCFM.forward_estimator already supports running it
through a TensorRT engine. This module builds that engine from the bundled
flow.decoder.estimator*.onnx and wraps it so it can be dropped in for the torch
estimator.

The released estimator ONNX has CFG batch 2 at its interface. Cross-request
flow batching needs CFG batch 2N. When requested, the builder makes the parsed
network inputs batch-dynamic and creates two optimization profiles: a dedicated
CFG=2 profile for the legacy path and a CFG=4..2N profile for batched flow.
The normal non-batched path retains the original static engine and cache key.

Precision: TensorRT >= 11 dropped the weakly-typed FP16/INT8 builder flags, so
fp16 only comes from a STRONGLY_TYPED network built from an fp16 ONNX. An fp32
ONNX is built fp32 + the TF32 matmul flag. EXPLICIT_BATCH is implicit.
"""

from __future__ import annotations

import os
import queue
import uuid

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import (
    _resolve_plan_path,
    _trt_logger,
)

logger = init_logger(__name__)

_DYNAMIC_INPUTS = ("x", "mask", "mu", "cond")
_ALL_INPUTS = ("x", "mask", "mu", "t", "spks", "cond")
_STATIC_CFG_BATCH = 2
_MIN_TIME = 4
_OPT_TIME = 500
_MAX_TIME = 3000
_DEFAULT_MAX_REQUEST_BATCH = 8
_BATCHED_OPT_CFG_BATCH = 8

# Original static-engine profile. Keep these values unchanged for the default
# COSYVOICE3_BATCH_FLOW=0 path.
_MIN_SHAPES = ((2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4))
_OPT_SHAPES = ((2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500))
_MAX_SHAPES = ((2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000))


def _is_fp16_onnx(onnx_path: str) -> bool:
    """Return whether this is the strongly-typed fp16 estimator export."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _profile_shapes(min_batch: int, opt_batch: int, max_batch: int):
    """Return TensorRT min/opt/max shapes for every estimator input."""

    def time_shapes(channels: int):
        return (
            (min_batch, channels, _MIN_TIME),
            (opt_batch, channels, _OPT_TIME),
            (max_batch, channels, _MAX_TIME),
        )

    return {
        "x": time_shapes(80),
        "mask": time_shapes(1),
        "mu": time_shapes(80),
        "t": ((min_batch,), (opt_batch,), (max_batch,)),
        "spks": ((min_batch, 80), (opt_batch, 80), (max_batch, 80)),
        "cond": time_shapes(80),
    }


def _dynamic_onnx_bytes(onnx_path: str) -> bytes:
    """Rewrite exported CFG batch=2 annotations to one symbolic dimension.

    The released ONNX carries batch=2 not only on the seven graph I/O values
    but also on intermediate value_info entries. TensorRT preserves those
    annotations during parsing, so changing only network inputs is insufficient
    for batch>2. Rewrite every tensor value whose leading dimension is the
    exported CFG batch and validate the resulting graph before building.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "CosyVoice3 TensorRT flow batching requires the 'onnx' package "
            "to rewrite the estimator's exported CFG batch metadata"
        ) from exc

    model = onnx.load(onnx_path)
    rewritten = 0
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape") or not tensor_type.shape.dim:
            continue
        batch_dim = tensor_type.shape.dim[0]
        if batch_dim.HasField("dim_value") and batch_dim.dim_value == _STATIC_CFG_BATCH:
            batch_dim.dim_param = "cfg_batch"
            rewritten += 1

    expected_io = set(_ALL_INPUTS)
    actual_inputs = {value.name for value in model.graph.input}
    missing = sorted(expected_io - actual_inputs)
    if missing:
        raise ValueError(f"Flow-estimator ONNX is missing expected inputs: {missing}")

    for value in list(model.graph.input) + list(model.graph.output):
        shape = value.type.tensor_type.shape
        if not shape.dim or shape.dim[0].dim_param != "cfg_batch":
            raise ValueError(f"Flow-estimator ONNX value {value.name!r} did not become batch-dynamic")

    onnx.checker.check_model(model)
    logger.info("Rewrote %d flow-estimator ONNX batch annotations to symbolic cfg_batch", rewritten)
    return model.SerializeToString()


def _add_dynamic_profile(builder, config, min_batch: int, opt_batch: int, max_batch: int) -> None:
    profile = builder.create_optimization_profile()
    for name, (mn, op, mx) in _profile_shapes(min_batch, opt_batch, max_batch).items():
        if profile.set_shape(name, mn, op, mx) is False:
            raise RuntimeError(f"TensorRT rejected optimization profile shape for {name}: {mn}, {op}, {mx}")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected flow-estimator optimization profile")


def _write_plan_atomically(engine_bytes, plan_path: str) -> None:
    tmp = f"{plan_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_created = False
    try:
        with open(tmp, "xb") as f:
            tmp_created = True
            f.write(engine_bytes)
        os.replace(tmp, plan_path)
        tmp_created = False
    except BaseException:
        if tmp_created:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Failed to remove temporary TensorRT plan %s", tmp, exc_info=True)
        raise


def _convert_onnx_to_trt(
    onnx_path: str,
    plan_path: str,
    strongly_typed: bool,
    *,
    dynamic_batch: bool = False,
    max_cfg_batch: int = _STATIC_CFG_BATCH,
) -> None:
    import tensorrt as trt

    logger.info(
        "Building flow-estimator TensorRT engine from %s (%s, dynamic_batch=%s) ...",
        onnx_path,
        "strongly-typed/fp16" if strongly_typed else "fp32+TF32",
        dynamic_batch,
    )
    trt_logger = _trt_logger()
    builder = trt.Builder(trt_logger)
    if strongly_typed:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:
        network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if dynamic_batch:
        onnx_bytes = _dynamic_onnx_bytes(onnx_path)
    else:
        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()
    if not parser.parse(onnx_bytes):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    workspace_bytes = 8 << 30 if dynamic_batch else 4 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if not strongly_typed:
        for _flag_name in ("FP16", "TF32"):
            _flag = getattr(trt.BuilderFlag, _flag_name, None)
            if _flag is not None:
                config.set_flag(_flag)
                break

    if dynamic_batch:
        if max_cfg_batch < 4 or max_cfg_batch % 2:
            raise ValueError(f"max_cfg_batch must be an even integer >= 4, got {max_cfg_batch}")
        # Profile 0 keeps the exact single-request specialization. Profile 1
        # serves request batches 2..N after CFG doubling.
        _add_dynamic_profile(builder, config, 2, 2, 2)
        opt_batch = min(_BATCHED_OPT_CFG_BATCH, max_cfg_batch)
        _add_dynamic_profile(builder, config, 4, opt_batch, max_cfg_batch)
    else:
        profile = builder.create_optimization_profile()
        for name, mn, op, mx in zip(_DYNAMIC_INPUTS, _MIN_SHAPES, _OPT_SHAPES, _MAX_SHAPES):
            profile.set_shape(name, mn, op, mx)
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    _write_plan_atomically(engine_bytes, plan_path)
    logger.info("Wrote flow-estimator TensorRT engine to %s", plan_path)


class TrtContextWrapper:
    """Pool of TensorRT execution contexts for the flow estimator."""

    def __init__(
        self,
        engine,
        device: str | torch.device,
        io_dtype: torch.dtype = torch.float32,
        trt_concurrent: int = 1,
        *,
        dynamic_batch: bool = False,
        max_cfg_batch: int = _STATIC_CFG_BATCH,
    ):
        self.trt_engine = engine
        self.io_dtype = io_dtype
        self.dynamic_batch = dynamic_batch
        self.max_cfg_batch = max_cfg_batch
        self._pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        for _ in range(trt_concurrent):
            ctx = engine.create_execution_context()
            assert ctx is not None, "failed to create TRT execution context (out of memory?)"
            stream = torch.cuda.Stream(torch.device(device))
            self._pool.put([ctx, stream])

    def _profile_index_for_batch(self, cfg_batch: int) -> int:
        if cfg_batch == _STATIC_CFG_BATCH:
            return 0
        if not self.dynamic_batch:
            raise ValueError(
                f"Static CosyVoice3 TensorRT estimator only supports CFG batch {_STATIC_CFG_BATCH}, got {cfg_batch}"
            )
        if cfg_batch < 4 or cfg_batch > self.max_cfg_batch or cfg_batch % 2:
            raise ValueError(
                f"Dynamic CosyVoice3 TensorRT estimator supports even CFG batches 4..{self.max_cfg_batch} "
                f"(plus batch 2), got {cfg_batch}"
            )
        return 1

    def prepare_context(self, context, stream, cfg_batch: int) -> int:
        """Select the optimization profile required by this CFG batch."""
        profile_index = self._profile_index_for_batch(cfg_batch)
        active_profile = int(getattr(context, "active_optimization_profile", 0))
        if active_profile != profile_index:
            result = context.set_optimization_profile_async(profile_index, stream.cuda_stream)
            if result is False:
                raise RuntimeError(
                    f"TensorRT failed to activate optimization profile {profile_index} for CFG batch {cfg_batch}"
                )
        return profile_index

    def acquire_estimator(self):
        return self._pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self._pool.put([context, stream])


def build_flow_estimator_trt(
    onnx_path: str,
    device: str | torch.device,
    *,
    dynamic_batch: bool = False,
    max_request_batch: int = _DEFAULT_MAX_REQUEST_BATCH,
) -> TrtContextWrapper:
    """Build/load the flow-estimator TRT engine and return a context-pool wrapper.

    With dynamic batching enabled, one engine carries a CFG=2 profile plus a
    CFG=4..2N profile. It uses a distinct cache key, so toggling batching never
    replaces the legacy static plan.
    """
    import tensorrt as trt

    strongly_typed = _is_fp16_onnx(onnx_path)
    max_cfg_batch = 2 * int(max_request_batch)
    if dynamic_batch and max_cfg_batch < 4:
        raise ValueError(f"max_request_batch must be >= 2 for dynamic batching, got {max_request_batch}")

    prefix = f"flow_estimator_cfg{max_cfg_batch}" if dynamic_batch else "flow_estimator"
    plan_path = _resolve_plan_path(onnx_path, prefix=prefix)
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(
            onnx_path,
            plan_path,
            strongly_typed=strongly_typed,
            dynamic_batch=dynamic_batch,
            max_cfg_batch=max_cfg_batch,
        )

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize flow-estimator TensorRT engine {plan_path}")
    logger.info(
        "Loaded flow-estimator TensorRT engine (%s, dynamic_batch=%s, max_cfg_batch=%d)",
        plan_path,
        dynamic_batch,
        max_cfg_batch,
    )
    io_dtype = torch.float16 if strongly_typed else torch.float32
    return TrtContextWrapper(
        engine,
        device=device,
        io_dtype=io_dtype,
        dynamic_batch=dynamic_batch,
        max_cfg_batch=max_cfg_batch if dynamic_batch else _STATIC_CFG_BATCH,
    )
