import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
random.seed(0)
import pathlib
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src.utils import context_util, html_util
from src import references as refs

data_dir = '../../bucket/wikipedia/1000docs_19513contexts_30maxtokens'
layers = [f'arr_{i}' for i in range(13)]
n_samples = 5
n_neighbors = 10
verbose = False

# For debugging
# layers = ['arr_0']
# n_samples = 1
# n_neighbors = 2

# Load contexts and acts
data_dir = os.path.abspath(data_dir)
corpus_contexts = pickle.load(open(os.path.join(data_dir, refs.contexts_fn), 'rb'))
corpus_acts = np.load(os.path.join(data_dir, refs.acts_fn))

# Nearest neighbor models - for facilitating fast nearest neighbor search
knn_models = {layer: NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(corpus_acts[layer]) for layer in layers}

# Find and save NNs
for context_idx in random.sample(range(len(corpus_contexts)), n_samples):  # random contexts
    context = corpus_contexts[context_idx]
    # For mapping each layer to neighbors in that layer
    neighbors_idxs = {}
    neighbors_htmls = {}
    for layer in layers:
        context_acts = corpus_acts[layer][context_idx]
        _, _neighbors_idxs = knn_models[layer].kneighbors([context_acts])
        _neighbors_idxs = _neighbors_idxs[0]  # drop empty dimension
        # Print neighbors
        if verbose:
            print(f'Layer {layer} \n' + '\n'.join([context_util.context_str(*corpus_contexts[idx]) for idx in _neighbors_idxs]))
        # Create extra fancy version of neighbors: get neighbors in html form, color-coded with some helpful info
        _neighbors_htmls = []
        for neighbor_idx in _neighbors_idxs:
            if corpus_contexts[neighbor_idx][0] == corpus_contexts[context_idx][0]:
                # color indicates a context in the same doc
                _neighbors_htmls.append(html_util.highlight(
                    context_util.context_html(*corpus_contexts[neighbor_idx]), color='lightsteelblue'))
            elif not any([neighbor_idx in neighbors_idxs[layer] for layer in layers]):
                # color indicates a new unseen neighbor
                _neighbors_htmls.append(html_util.highlight(
                    context_util.context_html(*corpus_contexts[neighbor_idx]), color='lightgreen'))
            else:
                _neighbors_htmls.append(f'{context_util.context_html(*corpus_contexts[neighbor_idx])}')
        neighbors_idxs[layer] = _neighbors_idxs
        neighbors_htmls[layer] = _neighbors_htmls
    # Save neighbors in html
    header_text = html_util.highlight(context_util.context_html(*context), color='lightsteelblue')
    layers_text = [f'Layer {layer}<br>' + '<br>'.join(neighbors_htmls[layer]) for layer in layers]
    with open(os.path.join(data_dir, f'nearest_neighbors_{context_idx}.html'), 'w') as f:
        f.write(f'<html><body> {header_text}<br><br><br>' + '<br><br><br>'.join(layers_text) + '</body></html>')