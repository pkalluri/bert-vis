"""Given a data directory containing an acts file (usually a large corpus), creates a pickle file containing,
in order, every layer's KNN model. These can be used downstream to get any vector's evolution of KNNs."""

import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import random
random.seed(0)
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src import references as refs


# Args
argparser = argparse.ArgumentParser()
argparser.add_argument('data_path', help='Data directory containing an activations file,'
                                         'and the generated KNN file will be put here.')
argparser.add_argument('-a', '--algorithm', default='ball_tree', help='KNN algorithm.')
args = argparser.parse_args()

# Prepping filenames
data_path = os.path.abspath(args.data_path)
acts_fn = refs.acts_fn
knn_models_fn = refs.knn_models_fn

# Build and save nearest neighbor models - for facilitating fast nearest neighbor search
acts = np.load(os.path.join(data_path, acts_fn))
with open(os.path.join(data_path, knn_models_fn), 'wb') as knn_models_file:
    for layer in acts.files:
        print(f'Modeling {layer}')
        knn_model = NearestNeighbors(algorithm=args.algorithm).fit(acts[layer])
        print('Writing')
        pickle.dump(knn_model, knn_models_file)
        del knn_model
