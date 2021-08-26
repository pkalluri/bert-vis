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
from src.utils import acts_util


# Args
argparser = argparse.ArgumentParser()
argparser.add_argument('data_path', help='Data directory containing an activations file,'
                                         'and the generated KNN file will be put here.')
argparser.add_argument('-a', '--algorithm', default='auto', help='which KNN algorithm.')
argparser.add_argument('-n', '--nneighbors', default=50, help='Set default number of neighbors to get when queried.')
argparser.add_argument('-o', '--output', default='', help='name of output file')
argparser.add_argument('-s', '--spherize', default=False, action='store_true', help='spherize acts to unit norm first?')
args = argparser.parse_args()


# Prepping filenames
data_path = os.path.abspath(args.data_path)
acts_fn = refs.acts_fn
knn_models_fn = args.output if args.output else refs.knn_models_fn

# Build and save nearest neighbor models - for facilitating fast nearest neighbor search
acts = np.load(os.path.join(data_path, acts_fn))
with open(os.path.join(data_path, knn_models_fn), 'wb') as knn_models_file:
    for layer in acts.files:
        _acts = acts[layer]
        if args.spherize:
            print('Spherizing')
            _acts = acts_util.spherize(_acts)
        print(f'Modeling {layer}')
        knn_model = NearestNeighbors(algorithm=args.algorithm, n_neighbors=args.nneighbors).fit(_acts)
        print('Writing')
        pickle.dump(knn_model, knn_models_file)
        del knn_model
