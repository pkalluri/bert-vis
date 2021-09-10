'''Given the path to a directory containing an acts file,
creates a subdirectory containing the corresponding knn models
(in the form of pickle files names '[layer name].pickle').'''

import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import click
import random
random.seed(0)

import os
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src.utils import references as refs
from src.utils import acts_util


@click.command(help='Given the path to a directory containing an acts file, ' \
    'creates a knn models subdirectory containing pickled knn models.',)
@click.argument('data_path')
def make_knn_models(data_path):
    data_path = os.path.abspath(data_path)
    acts_path = os.path.join(data_path, refs.acts_fn)
    models_path = os.path.join(data_path, refs.knn_models_dirname)
    os.mkdir(models_path)

    acts = np.load(acts_path)
    model = NearestNeighbors()
    for layer in acts:
        print(f'Fitting layer {layer}')
        _acts = acts[layer]
        model.fit(_acts)
        with open(os.path.join(models_path, f'{layer}.pickle'), 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    make_knn_models()
