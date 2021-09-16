"""Given a data directory containing many subdirectories, where each subdirectory contains an acts and contexts file,
consolidates all into one acts file and one contexts file."""
import numpy as np
import pickle
import glob
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src.utils import references as refs


argparser = argparse.ArgumentParser()
argparser.add_argument('data_dir', help='Directory containing many subdirectories, '
                                         'where each subdirectory contains one contexts file and one activations file)')
argparser.add_argument('out_data_dir', help='Directory in which to put generated contexts file and activations file.')
argparser.add_argument('-c', '--contexts', action='store_true', default=False, help='Consolidate contexts files?')
argparser.add_argument('-a', '--acts', action='store_true', default=False, help='Consolidate activations files?')
# argparser.add_argument('-l', '--layers', nargs='+', help='Which layers to consolidate')
args = argparser.parse_args()

contexts = [] if args.contexts else None
acts = {} if args.acts else None
first_subdir = True
for subdir in glob.iglob(os.path.join(args.data_dir, '*')):
    print(f'Found {subdir}.')
    if os.path.isdir(subdir):
        print(f'Loading {subdir}.')
        if args.contexts:
            contexts_path = glob.iglob(os.path.join(subdir, '*.pickle')).__next__()
            contexts.extend(pickle.load(open(contexts_path, 'rb')))

        if args.acts:
            acts_path = glob.iglob(os.path.join(subdir, '*.npz')).__next__()
            new_acts = np.load(acts_path)
            if first_subdir:
                acts.update(new_acts)
                layers = new_acts.files
                first_subdir = False
            else:
                for layer in layers:
                    acts[layer] = np.concatenate([acts[layer], new_acts[layer]])

print('Writing.')
if not os.path.exists(args.out_data_dir):
    os.mkdir(args.out_data_dir)
if args.contexts:
    pickle.dump(contexts, open(os.path.join(args.out_data_dir, refs.contexts_fn), 'wb'))
if args.acts:
    np.savez(os.path.join(args.out_data_dir, refs.acts_fn), **acts)
