# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.cosyvoice3 import flow_estimator_trt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _temporary_plans(plan_path: Path) -> list[Path]:
    return list(plan_path.parent.glob(f"{plan_path.name}.tmp.*"))


def test_write_plan_cleans_up_after_replace_failure(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    plan_path.write_bytes(b"existing plan")
    replace_error = OSError("replace failed")

    def fail_replace(source, destination):
        raise replace_error

    monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is replace_error
    assert plan_path.read_bytes() == b"existing plan"
    assert _temporary_plans(plan_path) == []


def test_write_plan_preserves_replace_error_when_cleanup_fails(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    replace_error = OSError("replace failed")

    def fail_replace(source, destination):
        raise replace_error

    def fail_unlink(path):
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)
    monkeypatch.setattr(flow_estimator_trt.os, "unlink", fail_unlink)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is replace_error
    assert len(_temporary_plans(plan_path)) == 1


def test_write_plan_cleans_up_after_write_failure(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    write_error = OSError("write failed")
    real_open = open

    class FailingWriter:
        def __init__(self, path, mode):
            self.file = real_open(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.file.close()

        def write(self, data):
            self.file.write(data[:1])
            raise write_error

    monkeypatch.setattr(flow_estimator_trt, "open", FailingWriter, raising=False)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is write_error
    assert not plan_path.exists()
    assert _temporary_plans(plan_path) == []


def test_write_plan_does_not_remove_a_colliding_temporary_file(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    token = "0" * 32
    temporary_path = Path(f"{plan_path}.tmp.{flow_estimator_trt.os.getpid()}.{token}")
    temporary_path.write_bytes(b"another writer")
    monkeypatch.setattr(flow_estimator_trt.uuid, "uuid4", lambda: flow_estimator_trt.uuid.UUID(hex=token))

    with pytest.raises(FileExistsError):
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert temporary_path.read_bytes() == b"another writer"
    assert not plan_path.exists()


def test_write_plan_supports_concurrent_publication(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    payloads = (b"a" * 4096, b"b" * 4096)
    barrier = threading.Barrier(len(payloads))
    source_paths = []
    source_paths_lock = threading.Lock()
    real_replace = flow_estimator_trt.os.replace

    def synchronized_replace(source, destination):
        with source_paths_lock:
            source_paths.append(Path(source))
        barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(flow_estimator_trt.os, "replace", synchronized_replace)

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [
            executor.submit(flow_estimator_trt._write_plan_atomically, payload, str(plan_path)) for payload in payloads
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(set(source_paths)) == len(payloads)
    assert plan_path.read_bytes() in payloads
    assert _temporary_plans(plan_path) == []



def test_dynamic_profile_shapes_cover_all_estimator_inputs():
    shapes = flow_estimator_trt._profile_shapes(4, 8, 16)

    assert set(shapes) == {"x", "mask", "mu", "t", "spks", "cond"}
    assert shapes["x"] == ((4, 80, 4), (8, 80, 500), (16, 80, 3000))
    assert shapes["mask"] == ((4, 1, 4), (8, 1, 500), (16, 1, 3000))
    assert shapes["t"] == ((4,), (8,), (16,))
    assert shapes["spks"] == ((4, 80), (8, 80), (16, 80))


def test_dynamic_onnx_rewrites_graph_and_internal_batch_metadata(tmp_path, monkeypatch):
    class FakeDim:
        def __init__(self, value=None, param=""):
            self.dim_value = value
            self.dim_param = param

        def HasField(self, name):
            return name == "dim_value" and self.dim_value is not None

    class FakeShape:
        def __init__(self, dims):
            self.dim = dims

    class FakeTensorType:
        def __init__(self, dims):
            self.shape = FakeShape(dims)

        def HasField(self, name):
            return name == "shape"

    class FakeType:
        def __init__(self, dims):
            self.tensor_type = FakeTensorType(dims)

    class FakeValue:
        def __init__(self, name, leading=2, tail=(80, 12)):
            dims = [FakeDim(leading)] + [FakeDim(value) for value in tail]
            self.name = name
            self.type = FakeType(dims)

    inputs = [
        FakeValue("x"),
        FakeValue("mask", tail=(1, 12)),
        FakeValue("mu"),
        FakeValue("t", tail=()),
        FakeValue("spks", tail=(80,)),
        FakeValue("cond"),
    ]
    output = FakeValue("estimator_out")
    internal = FakeValue("internal_activation")
    unrelated = FakeValue("constant_like", leading=1)

    class FakeGraph:
        input = inputs
        output = [output]
        value_info = [internal, unrelated]

    class FakeModel:
        graph = FakeGraph()

        @staticmethod
        def SerializeToString():
            return b"rewritten-model"

    class FakeChecker:
        checked = False

        @classmethod
        def check_model(cls, model):
            assert model is FakeModel()
            cls.checked = True

    class FakeOnnx:
        checker = FakeChecker

        @staticmethod
        def load(path):
            assert path == str(tmp_path / "estimator.onnx")
            return FakeModel()

    monkeypatch.setitem(__import__("sys").modules, "onnx", FakeOnnx)
    result = flow_estimator_trt._dynamic_onnx_bytes(str(tmp_path / "estimator.onnx"))

    assert result == b"rewritten-model"
    assert FakeChecker.checked
    for value in inputs + [output, internal]:
        assert value.type.tensor_type.shape.dim[0].dim_param == "cfg_batch"
    assert unrelated.type.tensor_type.shape.dim[0].dim_param == ""


def test_dynamic_onnx_rejects_missing_expected_input(tmp_path, monkeypatch):
    class FakeDim:
        dim_value = 2
        dim_param = ""

        @staticmethod
        def HasField(name):
            return name == "dim_value"

    class FakeTensorType:
        shape = type("Shape", (), {"dim": [FakeDim()]})()

        @staticmethod
        def HasField(name):
            return name == "shape"

    class FakeValue:
        def __init__(self, name):
            self.name = name
            self.type = type("Type", (), {"tensor_type": FakeTensorType()})()

    class FakeModel:
        graph = type(
            "Graph",
            (),
            {
                "input": [FakeValue(name) for name in ("x", "mask", "mu", "t", "spks")],
                "output": [FakeValue("estimator_out")],
                "value_info": [],
            },
        )()

    class FakeOnnx:
        checker = type("Checker", (), {"check_model": staticmethod(lambda model: None)})

        @staticmethod
        def load(path):
            return FakeModel()

    monkeypatch.setitem(__import__("sys").modules, "onnx", FakeOnnx)

    with pytest.raises(ValueError, match="missing expected inputs"):
        flow_estimator_trt._dynamic_onnx_bytes(str(tmp_path / "estimator.onnx"))


def test_context_wrapper_switches_between_static_and_batched_profiles():
    wrapper = object.__new__(flow_estimator_trt.TrtContextWrapper)
    wrapper.dynamic_batch = True
    wrapper.max_cfg_batch = 16

    class FakeContext:
        def __init__(self):
            self.active_optimization_profile = 0
            self.calls = []

        def set_optimization_profile_async(self, profile_index, stream):
            self.calls.append((profile_index, stream))
            self.active_optimization_profile = profile_index
            return True

    class FakeStream:
        cuda_stream = 12345

    context = FakeContext()
    stream = FakeStream()

    assert wrapper.prepare_context(context, stream, 2) == 0
    assert context.calls == []

    assert wrapper.prepare_context(context, stream, 4) == 1
    assert context.calls == [(1, 12345)]

    assert wrapper.prepare_context(context, stream, 8) == 1
    assert context.calls == [(1, 12345)]

    assert wrapper.prepare_context(context, stream, 2) == 0
    assert context.calls == [(1, 12345), (0, 12345)]


@pytest.mark.parametrize("cfg_batch", [1, 3, 17, 18])
def test_context_wrapper_rejects_unsupported_dynamic_cfg_batch(cfg_batch):
    wrapper = object.__new__(flow_estimator_trt.TrtContextWrapper)
    wrapper.dynamic_batch = True
    wrapper.max_cfg_batch = 16

    with pytest.raises(ValueError, match="supports even CFG batches"):
        wrapper._profile_index_for_batch(cfg_batch)


def test_static_context_wrapper_keeps_cfg2_contract():
    wrapper = object.__new__(flow_estimator_trt.TrtContextWrapper)
    wrapper.dynamic_batch = False
    wrapper.max_cfg_batch = 2

    assert wrapper._profile_index_for_batch(2) == 0
    with pytest.raises(ValueError, match="only supports CFG batch 2"):
        wrapper._profile_index_for_batch(4)
