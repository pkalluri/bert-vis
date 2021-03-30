from lucid.misc.channel_reducer import ChannelReducer
import numpy as np


def reduce_activations(acts: np.ndarray, reduction: str = 'NMF', dim: int = 6) -> np.ndarray:
    """
    Given activations, perform the specified dimensionality reduction.

    Returns: Array of shape (LENGTH OF ACTS, DIM)
    """
    reducer = ChannelReducer(dim, reduction)
    if reduction == 'NMF':
        # NMF requires activations to be positive
        acts = get_positive_activations(acts)
    return reducer._reducer.fit_transform(acts)


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
