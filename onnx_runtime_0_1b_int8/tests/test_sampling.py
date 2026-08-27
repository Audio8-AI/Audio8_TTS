from __future__ import annotations

import numpy as np

from arktts_runtime.runtime import _sample


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
