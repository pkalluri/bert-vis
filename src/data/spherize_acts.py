import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from src.utils import references as refs
from src.utils.acts_util import spherize

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='Path to directory containing input acts file.')
parser.add_argument('out_data_dir', help='Path to directory to put output spherized acts file.')
args = parser.parse_args()

acts = np.load(os.path.join(args.data_dir, refs.acts_fn))
layers = acts.files
normalized_acts = {layer: spherize(acts[layer]) for layer in layers}

if not os.path.exists(args.out_data_dir):
    os.mkdir(args.out_data_dir)
np.savez(os.path.join(args.out_data_dir, refs.acts_fn), **normalized_acts)