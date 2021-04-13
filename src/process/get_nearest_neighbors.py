import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
random.seed(0)
import pathlib
import sys
parent_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, parent_path)
from src.utils import context_util, html_util

data_dir = '../../bucket/wikipedia/1000docs_19513contexts_30maxtokens'
contexts_filename = 'contexts.pickle'
acts_filename = 'activations.npz'
layers = [f'arr_{i}' for i in range(13)]
n_samples = 5
n_neighbors = 10
verbose = False
# For debugging
# layers = ['arr_0']
# n_samples = 1
# n_neighbors = 2

# Load contexts and acts
with open(os.path.join(os.path.abspath(data_dir), contexts_filename), 'rb') as f:
    contexts = pickle.load(f)
acts = np.load(os.path.join(data_dir, acts_filename))

# Nearest neighbor models - for facilitating fast nearest neighbor search
nearest_neighbor_models = {layer: NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(acts[layer]) for layer in layers}

# Find and save NNs
for context_idx in random.sample(range(len(contexts)), n_samples):  # random contexts
    context = contexts[context_idx]
    # For mapping each layer to neighbors in that layer
    layer_to_neighbors_idxs = {}
    layer_to_neighbors_htmls = {}
    for layer in layers:
        context_acts = acts[layer][context_idx]
        _, neighbors_indices = nearest_neighbor_models[layer].kneighbors([context_acts])
        neighbors_indices = neighbors_indices[0]  # drop empty dimension
        # Print neighbors
        if verbose:
            print(f'Layer {layer} \n' + '\n'.join([context_util.context_str(*contexts[idx]) for idx in neighbors_indices]))
        # Create extra fancy version of neighbors: get neighbors in html form, color-coded with some helpful info
        neighbors_htmls = []
        for neighbor_idx in neighbors_indices:
            if contexts[neighbor_idx][0] == contexts[context_idx][0]:
                # color indicates a context in the same doc
                neighbors_htmls.append(html_util.highlight_html(
                    context_util.context_html(*contexts[neighbor_idx]), color='lightsteelblue'))
            elif not any([neighbor_idx in layer_to_neighbors_idxs[layer] for layer in layer_to_neighbors_idxs]):
                # color indicates a new unseen neighbor
                neighbors_htmls.append(html_util.highlight_html(
                    context_util.context_html(*contexts[neighbor_idx]), color='lightgreen'))
            else:
                neighbors_htmls.append(f'{context_util.context_html(*contexts[neighbor_idx])}')
        layer_to_neighbors_idxs[layer] = neighbors_indices
        layer_to_neighbors_htmls[layer] = neighbors_htmls
    # Save neighbors in html
    header_text = html_util.highlight_html(context_util.context_html(*context), color='lightsteelblue')
    layers_text = [f'Layer {layer}<br>' +'<br>'.join(layer_to_neighbors_htmls[layer]) for layer in layers]
    with open(os.path.join(data_dir, f'nearest_neighbors_{context_idx}.html'), 'w') as f:
        f.write(f'<html><body> {header_text}<br><br><br>' + '<br><br><br>'.join(layers_text) + '</body></html>')