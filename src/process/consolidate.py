"""Given a dataset directory of text files and activations,
reads and consolidates these into one contexts file and one activations file."""
import glob
import os
import numpy as np
import pickle
import random
random.seed(0)
import pathlib
import sys
parent_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, parent_path)
from src import references

# Parameters
data_path = 'bucket/wikipedia'
max_docs = None  # Max number of documents to read or None. If None, this is ignored.
max_contexts = None  # Max number of contexts to read or None. If None, this is ignored.
max_toks = 50  # Max number of tokens in an acceptable document. If None, this is ignored.

# Get all contexts and acts
toks_paths = glob.iglob(data_path + '/**/*.pickle', recursive=True)
if max_docs:  # Limit to max number of docs
    toks_paths = list(toks_paths)
    random.shuffle(toks_paths)
# Create a list of contexts. Each context will be a tuple: (doc's tokens, position in doc).
contexts = []
# Create a dictionary to map layer to list of docs' activations.
# Each doc's activations will be size (# contexts x size of embedding)
layers = {}
n_docs_consolidated = 0
n_long_docs = 0
for toks_path in toks_paths:
    if os.path.split(toks_path)[-1].startswith(references.CONTEXTS_BASENAME):
        print(f'Skipping {references.CONTEXTS_BASENAME} path: ' + toks_path)
        continue  # Skip
    with open(toks_path, 'rb') as f:
        toks = pickle.load(f)
    if max_toks and len(toks) > max_toks:
        n_long_docs += 1
        print('Skipping long doc')
        continue  # Skip
    if max_contexts and max_contexts < len(contexts) + len(toks):
        break  # Done
    # Add new contexts
    for tok_i in range(len(toks)):
        context = (toks, tok_i)
        contexts.append(context)
    # Add new acts
    acts_path = toks_path.replace(references.TOKENS_BASENAME, references.ACTIVATIONS_BASENAME).replace(references.TOKENS_EXTENSION, references.ACTIVATIONS_EXTENSION)
    acts = np.load(acts_path)
    for layer in acts.files:
        if layer not in layers:
            layers[layer] = acts[layer]
        else:
            layers[layer] = np.concatenate([layers[layer], acts[layer]])
    print(f'Doc {n_docs_consolidated}: {toks_path} ({len(toks)} tokens) --> {len(contexts)} total contexts')
    n_docs_consolidated += 1
    if n_docs_consolidated == max_docs:
        break  # Done

print(f'Found {n_docs_consolidated} docs & {len(contexts)} contexts and obtained activations of shape {layers[layer].shape}')
if max_toks:
    print(f'Ignored {n_long_docs} docs longer than {max_toks} tokens.')
out_dir_path = os.path.join(
    data_path,
    f'{n_docs_consolidated}docs_{len(contexts)}contexts_{max_toks}maxtokens'
)
os.mkdir(out_dir_path)
consolidated_contexts_fn = f'{references.CONTEXTS_BASENAME}.pickle'
consolidated_acts_fn = f'{references.ACTIVATIONS_BASENAME}.npz'
with open(os.path.join(out_dir_path, consolidated_contexts_fn), 'wb') as f:
    pickle.dump(contexts, f)
np.savez(os.path.join(out_dir_path, consolidated_acts_fn), **layers)
