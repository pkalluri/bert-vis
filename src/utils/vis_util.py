import math
import numpy as np


def hue_to_rgb(ang, warp=True):
    """
    Produce an RGB unit vector corresponding to a hue of a given angle.
    Code adapted from Tensorflow Lucid library.
    """
    ang = ang - 360 * (ang // 360)
    colors = np.asarray([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
    ])
    colors = colors / np.linalg.norm(colors, axis=1, keepdims=True)
    R = 360 / len(colors)
    n = math.floor(ang / R)
    D = (ang - n * R) / R

    if warp:
        # warping the angle away from the primary colors (RGB)
        # helps make equally-spaced angles more visually distinguishable
        adj = lambda x: math.sin(x * math.pi / 2)
        if n % 2 == 0:
            D = adj(D)
        else:
            D = 1 - adj(1 - D)

    v = (1 - D) * colors[n] + D * colors[(n + 1) % len(colors)]
    return v / np.linalg.norm(v)


def channels_to_rgb(X):
    """
    Convert n-dimensional values to representative 3-dimensional RGB values.
    Code adapted from Tensorflow Lucid library.
    """
    if (X < 0).any():
        X = np.concatenate([np.maximum(0, X), np.maximum(0, -X)], axis=-1)

    K = X.shape[-1]
    rgb = 0
    for i in range(K):
        ang = 360 * i / K  # angle of the ith point
        color = hue_to_rgb(ang)  # color of the ith point
        color = color[tuple(None for _ in range(1))]  # reshape if necessary
        # convert the ith column (list of values) into a list of vectors all in the chosen direction
        rgb += X[..., i, None] * color
    # We now have 3D vectors, but we need to normalize them into RGBs
    ones_col = np.ones(X.shape[:-1])[..., None]  # a list of ones
    # rgb += ones_col * (X.sum(-1) - X.max(-1))[..., None]
    rgb /= 1e-4 + np.linalg.norm(rgb, axis=-1, keepdims=True)  # normalize so each 3-dim vector is unit length
    return rgb
