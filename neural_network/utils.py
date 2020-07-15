from typing import Tuple

import numpy as np


def make_parabolas(n_samples: int = 1000, noise: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Generate classification task data in form of two parabolas.

    :param n_samples: number of samples
    :param noise: std. deviation of noise
    :return: objects and their labels
    """

    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    x = []
    y = []

    for i, n in enumerate([n_neg, n_pos]):
        ox = np.random.uniform(-1, 3, size=n)
        oy = i + ox ** 2 + np.random.normal(scale=noise, size=n)

        x_ = np.vstack([ox, oy]).T
        y_ = np.ones(n) * i

        x.append(x_)
        y.append(y_)

    x = np.vstack(x)
    y = np.hstack(y).astype(int)

    return x, y
