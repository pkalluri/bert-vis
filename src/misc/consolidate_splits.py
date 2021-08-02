"""Given a data directory containing many subdirectories, where each subdirectory contains an acts and contexts file,
consolidates all into one acts file and one contexts file."""
import numpy as np
import pickle
import glob
import os
import argparse
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src import references as refs


argparser = argparse.ArgumentParser()
argparser.add_argument('data_path', help='Dataset directory (directory should contain many subdirectories, where each subdirectory contains one contexts and one activations file)')
argparser.add_argument('-c', '--contexts', action='store_true', default=False, help='Consolidate contexts files?')
argparser.add_argument('-a', '--acts', action='store_true', default=False, help='Consolidate activations files?')
argparser.add_argument('-l', '--layers', nargs='+', help='Which layers to consolidate')
args = argparser.parse_args()

corpus_contexts = [] if args.contexts else None
corpus_acts = {} if args.acts else None
first_subdir = True
for subdir in glob.iglob(os.path.join(args.data_path, '*')):
    print(f'Found {subdir}.')
    if os.path.isdir(subdir) and 'docs' in subdir:
        print(f'Loading {subdir}.')
        if args.contexts:
            pickle_path = glob.iglob(os.path.join(subdir, '*.pickle')).__next__()
            new_contexts = pickle.load(open(pickle_path, 'rb'))
            corpus_contexts.extend(new_contexts)

        if args.acts:
            npz_path = glob.iglob(os.path.join(subdir, '*.npz')).__next__()
            new_acts = np.load(npz_path)
            if first_subdir:
                corpus_acts.update(new_acts)
                layers = new_acts.files
                first_subdir = False
            else:
                for layer in layers:
                    corpus_acts[layer] = np.concatenate([corpus_acts[layer], new_acts[layer]])

        print('writing')
        if args.contexts:
            pickle.dump(corpus_contexts, open(os.path.join(args.data_path, refs.contexts_fn), 'wb'))
            print(len(corpus_contexts))
        if args.acts:
            np.savez(os.path.join(args.data_path, refs.acts_fn), **corpus_acts)
            print(corpus_acts[layers[0]].shape)
    else:
        print(f'Skipping {subdir}.')
