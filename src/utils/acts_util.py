from lucid.misc.channel_reducer import ChannelReducer
from sklearn.manifold import TSNE
import numpy as np


def fit_components(acts: np.ndarray, reduction: str = 'NMF', dim: int = 6) -> (np.ndarray, np.ndarray):
    """
    Given activations, perform the specified dimensionality reduction.

    Returns: Array of shape (DIM, DIM OF ACTS), Array of shape (LENGTH OF ACTS, DIM)
    """
    reducer = ChannelReducer(dim, reduction)
    if reduction == 'NMF':  # NMF requires activations to be positive
        acts = get_positive_activations(acts)
    reduced_acts = reducer._reducer.fit_transform(acts)
    components = reducer._reducer.components_
    if reduction == 'NMF':  # positive NMF components must be turned back into normal activations
        components = reverse_get_positive_activations(components)
    return components, reduced_acts


def fit_reducer(acts: np.ndarray, reduction: str = 'NMF', dim: int = 6):
    """
    Given activations, fit the specified dimensionality reduction model.

    Returns: Fit reduction model, Array of shape (LENGTH OF ACTS, DIM)
    """
    if reduction == 'TSNE':  # neighbor-based
        reducer = TSNE(n_components=dim)
        reduced_acts = reducer.fit_transform(acts)
    else:  # decomposition approach
        reducer = ChannelReducer(dim, reduction)
        if reduction == 'NMF':  # NMF requires activations to be positive
            acts = get_positive_activations(acts)
        reduced_acts = reducer._reducer.fit_transform(acts)
    return reducer, reduced_acts


def get_positive_activations(acts: np.ndarray) -> np.ndarray:
    """
    Return a twice-as-long positive array instead,
    with the originally positive values in the first half and the originally negative values in the second half.
    Essentially, this contains all the information in the original array, but in the form of a positive array.
    e.g. [-1, 2, 3] -> [0, 2, 3, 1, 0, 0]
    """
    return np.concatenate([np.maximum(0, acts), np.maximum(-acts, 0)], axis=-1)


def reverse_get_positive_activations(acts: np.ndarray) -> np.ndarray:
    """
    Reverses the get_positive_activations function.
    The twice-as-long positive array is turned back into regular-size array with positive and negative values.
    e.g. [0, 2, 3, 1, 0, 0] -> [-1, 2, 3]
    """
    curr_acts_dim = len(acts[0])
    destination_dim = int(curr_acts_dim/2)
    return acts[:, :destination_dim] + -1 * acts[:, destination_dim:]


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


