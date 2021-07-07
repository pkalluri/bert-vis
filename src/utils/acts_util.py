from lucid.misc.channel_reducer import ChannelReducer
from sklearn.manifold import TSNE
import numpy as np


def reduce_activations(acts: np.ndarray, reduction: str = 'NMF', dim: int = 6) -> np.ndarray:
    """
    Given activations, perform the specified dimensionality reduction.

    Returns: Array of shape (LENGTH OF ACTS, DIM)
    """
    if reduction == 'TSNE':  # neighbor-based
        reducer = TSNE(n_components=dim)
        return reducer.fit_transform(acts)
    else:  # decomposition approach
        reducer = ChannelReducer(dim, reduction)
        if reduction == 'NMF':  # NMF requires activations to be positive
            acts = get_positive_activations(acts)
        return reducer._reducer.fit_transform(acts)


def fit_reducer(acts: np.ndarray, reduction: str = 'NMF', dim: int = 6):
    """
    Given activations, fit the specified dimensionality reduction model.

    Returns: Fit reduction model, Array of shape (LENGTH OF ACTS, DIM)
    """
    if reduction == 'TSNE':  # neighbor-based
        reducer = TSNE(n_components=dim)
        reduced_acts = reducer.fit_transform(acts)
        return reducer, reduced_acts
    else:  # decomposition approach
        reducer = ChannelReducer(dim, reduction)
        if reduction == 'NMF':  # NMF requires activations to be positive
            acts = get_positive_activations(acts)
        reduced_acts = reducer._reducer.fit_transform(acts)
        return reducer._reducer, reduced_acts


def get_positive_activations(acts: np.ndarray) -> np.ndarray:
    """
    If any activations are negative, return a twice-as-long positive array instead,
    with the originally positive values in the first half and the originally negative values in the second half.
    Essentially, this contains all the information in the original array, but in the form of a positive array.
    e.g. [-1, 2, 3] -> [0, 2, 3, 1, 0, 0]
    """
    if (acts > 0).all():
        return acts
    else:
        return np.concatenate([np.maximum(0, acts), np.maximum(-acts, 0)], axis=-1)


def mean_center(acts: np.ndarray) -> np.ndarray:
    """Return mean-centered activations (such that the mean activation is now at the origin)."""
    return acts - np.mean(acts, axis=0)


def normalize(acts: np.ndarray, mean_centered=False) -> np.ndarray:
    """
    Return normalized activations, such that
    if not mean-centered, the magnitude of activations is between 0 and 1,
    and if mean-centered, the magnitude of activations is between 0 and 1.
    """
    if mean_centered:
        acts = acts - np.nanmean(acts)
    else:
        acts = acts - np.nanmin(acts)
    largest_magnitude = np.nanmax(np.linalg.norm(acts))
    return acts/largest_magnitude
