"""Given a data directory containing an acts file, creates a pickle file containing,
in order, every layer's KNN model. These can be used downstream to get any vector's evolution of KNNs"""

import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import random
random.seed(0)
import pathlib
import sys
parent_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, parent_path)


# Args
argparser = argparse.ArgumentParser()
argparser.add_argument('data', help='Data directory containing one contexts and one activations file.')
argparser.add_argument('-n', '--neighbors', type=int, help='Number of neighbors.')
args = argparser.parse_args()


# Prepping filenames
data_path = os.path.abspath(args.data)
contexts_filename = 'contexts.pickle'
acts_filename = 'activations.npz'
knn_models_filename = f'KNN_models_K{args.neighbors}.pickle'


# Load contexts and acts
with open(os.path.join(data_path, contexts_filename), 'rb') as f:
    contexts = pickle.load(f)
acts = np.load(os.path.join(data_path, acts_filename))


# Build and save nearest neighbor models - for facilitating fast nearest neighbor search
knn_models_path = os.path.join(data_path, knn_models_filename)
with open(knn_models_path, 'wb') as knn_models_file:
    for layer in acts.files:
        print(f'Modeling {layer}')
        knn_model = NearestNeighbors(n_neighbors=args.neighbors, algorithm='ball_tree').fit(acts[layer])
        print('Writing')
        pickle.dump(knn_model, knn_models_file)
