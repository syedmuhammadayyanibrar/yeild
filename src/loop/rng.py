import numpy as np

from loop.config import LOOP_SEED


def get_rng(seed: int | None = None) -> np.random.Generator:
    if seed is None:
        seed = LOOP_SEED
    return np.random.default_rng(seed)
