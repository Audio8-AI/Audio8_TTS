from __future__ import annotations

import numpy as np

from arktts_runtime.runtime import ArkTtsRuntime, _sample


def test_sample_uses_gumbel_max_noise() -> None:
    # This seed distinguishes Gumbel-max from the tempting but incorrect -log(U)
    # shortcut, which can bias sampling toward EOS and truncate speech.
    result = _sample(
        np.asarray([0.0, 1.0], dtype=np.float32),
        temperature=1.0,
        top_p=1.0,
        top_k=2,
        rng=np.random.default_rng(0),
    )
    assert result == 0


def test_first_semantic_step_cannot_return_eos() -> None:
    runtime = object.__new__(ArkTtsRuntime)
    runtime.manifest = {
        "codebook_size": 2,
        "semantic_begin_id": 100,
        "semantic_end_id": 101,
        "im_end_id": 4096,
        "slow_logits_layout": "relative_semantic_then_eos",
    }
    logits = np.asarray([0.0, 1.0, 100.0], dtype=np.float32)

    first = runtime._sample_semantic(
        logits,
        previous=[],
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        rng=np.random.default_rng(0),
    )
    assert first == 101

    later = runtime._sample_semantic(
        logits,
        previous=[first],
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        rng=np.random.default_rng(0),
    )
    assert later == 4096
