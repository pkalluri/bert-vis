"""Given a data directory containing many subdirectories, where each subdirectory contains an acts and contexts file, consolidates all into
one acts file and one contexts file."""
import numpy as np
import pickle
import glob
import os
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('data')
argparser.add_argument('-c', '--contexts', action='store_true', default=False)
contexts_path = 'contexts.pickle'
argparser.add_argument('-a', '--acts', action='store_true', default=False)
acts_path = 'activations.npz'
argparser.add_argument('-l', '--layers', nargs='+')
args = argparser.parse_args()

contexts = [] if args.contexts else None
layer_to_acts = {} if args.acts else None
first_subdir = True
for subdir in glob.iglob(os.path.join(args.data, '*')):
    print(f'Found {subdir}.')
    if os.path.isdir(subdir) and 'docs' in subdir:
        print(f'Loading {subdir}.')
        if args.contexts:
            pickle_path = glob.iglob(os.path.join(subdir, '*.pickle')).__next__()
            with open(pickle_path, 'rb') as f:
                new_contexts = pickle.load(f)
            contexts.extend(new_contexts)

        if args.acts:
            npz_path = glob.iglob(os.path.join(subdir, '*.npz')).__next__()
            layer_to_new_acts = np.load(npz_path)
            if first_subdir:
                layer_to_acts.update(layer_to_new_acts)
                layers = layer_to_new_acts.files
                first_subdir = False
            else:
                for layer in layers:
                    layer_to_acts[layer] = np.concatenate([layer_to_acts[layer], layer_to_new_acts[layer]])

        print('writing')
        if args.contexts:
            pickle.dump(contexts, open(os.path.join(args.data, contexts_path), 'wb'))
            print(len(contexts))
        if args.acts:
            np.savez(os.path.join(args.data, acts_path), **layer_to_acts)
            print(layer_to_acts[layers[0]].shape)
    else:
        print(f'Skipping {subdir}.')