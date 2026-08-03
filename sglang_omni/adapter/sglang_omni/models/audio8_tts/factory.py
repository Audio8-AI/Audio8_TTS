# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from sglang_omni.engines.omni.engine import OmniEngine
from sglang_omni.engines.omni.scheduler import Scheduler
from sglang_omni.models.audio8_tts.attention_backend import (
    ATTENTION_BACKEND_ENV,
    PORTABLE_ATTENTION_BACKEND,
    fa3_kernels_available,
)
from sglang_omni.models.audio8_tts.runtime.audio8_sglang_ar import (
    Audio8IterationController,
    Audio8ModelRunner,
    Audio8ResourceManager,
)


def create_audio8_engine(
    server_args: Any,
    *,
    gpu_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    stream_fn: Callable[[str, Any], None] | None = None,
) -> OmniEngine:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangBatchPlanner
    from sglang_omni.models.audio8_tts.sglang_model import Audio8SGLangModel

    # Register lazily in the worker process so the adapter does not require a
    # patch to SGLang-Omni's hard-coded model registry bootstrap.
    ModelRegistry.models["ArkttsModel"] = Audio8SGLangModel

    if server_args.attention_backend is None:
        server_args.attention_backend = "fa3"
    disable_attr = "disable_" + "cuda_graph"
    want_cuda_graph = not bool(getattr(server_args, disable_attr, False))
    setattr(server_args, disable_attr, True)
    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )
    model_worker.model_runner.model.setup_audio8_decode(
        server_args.max_running_requests
    )
    setattr(server_args, disable_attr, not want_cuda_graph)
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    request_pool, kv_allocator = model_worker.get_memory_pool()
    tree_cache = create_tree_cache(
        server_args,
        request_pool,
        kv_allocator,
        server_args.page_size,
    )
    prefill_manager = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=kv_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=False,
    )
    decode_manager = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=kv_allocator,
        on_retract=lambda request: prefill_manager.add_one_request(request),
    )
    planner = SGLangBatchPlanner(prefill_manager, decode_manager, server_args)
    def stream_adapter(request: Any, output: Any) -> Any:
        if output.data is None:
            return None
        codes = output.data.codes.clone()
        if stream_fn is not None:
            stream_fn(request.request_id, codes)
        return codes

    scheduler = Scheduler(
        batch_planner=planner,
        resource_manager=Audio8ResourceManager(
            kv_allocator,
            request_pool,
            tree_cache,
        ),
        iteration_controller=Audio8IterationController(
            tree_cache,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        ),
        stream_adapter=stream_adapter,
    )
    return OmniEngine(
        scheduler=scheduler,
        model_runner=Audio8ModelRunner(model_worker, planner),
        enable_overlap=False,
    )


def make_server_args(model_path: str) -> Any:
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        mem_fraction_static=float(os.getenv("AUDIO8_TTS_MEM_FRACTION_STATIC", "0.2")),
        chunked_prefill_size=int(os.getenv("AUDIO8_TTS_CHUNKED_PREFILL_SIZE", "2048")),
        max_running_requests=int(os.getenv("AUDIO8_TTS_MAX_RUNNING_REQUESTS", "32")),
        disable_radix_cache=os.getenv("AUDIO8_TTS_DISABLE_RADIX_CACHE", "1") == "1",
    )
    setattr(
        server_args,
        "disable_" + "cuda_graph",
        os.getenv("AUDIO8_TTS_DISABLE_CUDA_GRAPH", "0") == "1",
    )
    server_args.enable_torch_compile = (
        os.getenv("AUDIO8_TTS_ENABLE_TORCH_COMPILE", "0") == "1"
    )
    torch_compile_max_bs = os.getenv("AUDIO8_TTS_TORCH_COMPILE_MAX_BS")
    if torch_compile_max_bs is not None:
        server_args.torch_compile_max_bs = int(torch_compile_max_bs)
    attention_backend = os.getenv(ATTENTION_BACKEND_ENV)
    if attention_backend is None and not fa3_kernels_available():
        # No FA3 kernel image on this device; pick the portable backend rather
        # than leaving SGLang to fail at kernel launch.
        attention_backend = PORTABLE_ATTENTION_BACKEND
    if attention_backend:
        server_args.attention_backend = attention_backend
    return server_args
