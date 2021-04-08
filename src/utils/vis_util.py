import math
import numpy as np
from bokeh.palettes import Inferno, Category10, Category20, Category20c, Pastel1, Pastel2, Bokeh, Plasma


def hue_to_rgb(ang, warp=True):
    """
    Given a hue of a particular angle, produces a corresponding RGB unit vector.
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


def channels_to_rgbs(X):
    """
    Given n-dimensional vectors, converts to representative 3-dimensional RGB values.
    In particular, the n dimensions are mapped to n colors,
    and each n-dimensional vector is converted to the corresponding linear combination of the n colors.
    Code adapted from Tensorflow Lucid library.
    """
    if (X < 0).any():
        X = np.concatenate([np.maximum(0, X), np.maximum(0, -X)], axis=-1)

    K = X.shape[-1]
    rgbs = 0
    for i in range(K):
        ang = 360 * i / K  # angle of the ith point
        color = hue_to_rgb(ang)  # color of the ith point
        color = color[tuple(None for _ in range(1))]  # reshape if necessary
        # convert the ith column (list of values) into a list of vectors all in the chosen direction
        rgbs += X[..., i, None] * color
    # We now have 3D vectors, but we need to normalize them into RGBs
    rgbs /= 1e-4 + np.linalg.norm(rgbs, axis=-1, keepdims=True)  # normalize so each 3-dim vector is unit length
    rgbs *= 255
    return rgbs.astype(int)


BIG_PALETTE = Plasma[256]


def categorical_list_to_color_list(categorical_list, palette=BIG_PALETTE, reverse_palette=False):
    if reverse_palette:
        palette = palette[::-1]
    categories = set(categorical_list)
    n_colors_per_category = len(palette) // (len(categories) - 1)
    sampled_palette = palette[::n_colors_per_category] + (palette[-1],)
    categories_to_colors = {category: sampled_palette[i] for i, category in enumerate(categories)}
    return [categories_to_colors[elt] for elt in categorical_list]
