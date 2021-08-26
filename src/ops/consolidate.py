"""Given a dataset directory of text files and activations,
reads and consolidates these into one contexts file and one activations file."""
import glob
import os
import numpy as np
import pickle
import random
random.seed(0)
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src import references as refs


# Parameters
data_path = 'bucket/wikipedia'
max_docs = None  # Max number of documents to read or None. If None, this is ignored.
max_contexts = None  # Max number of contexts to read or None. If None, this is ignored.
max_toks = 50  # Max number of tokens in an acceptable document. If None, this is ignored.

# Get all contexts and acts
docs_paths = glob.iglob(data_path + '/**/*.pickle', recursive=True)
if max_docs:  # Limit to max number of docs
    docs_paths = list(docs_paths)
    random.shuffle(docs_paths)
# Create a list of contexts. Each context will be a tuple: (doc's tokens, position in doc).
corpus_contexts = []
# Create a dictionary to map layer to list of docs' activations.
# Each doc's activations will be size (# contexts x size of embedding)
corpus_acts = {}
n_docs_consolidated = 0
n_long_docs = 0
for doc_path in docs_paths:
    toks = pickle.load(open(doc_path, 'rb'))
    if max_toks and len(toks) > max_toks:
        n_long_docs += 1
        print('Skipping long doc')
        continue  # Skip this doc
    elif max_contexts and len(corpus_contexts) >= max_contexts:
        break  # Done with all docs
    else:
        # Add new contexts
        for pos in range(len(toks)):
            context = (toks, pos)
            corpus_contexts.append(context)
        # Add new acts
        acts_path = doc_path.replace(refs.toks_fn, refs.acts_fn)
        acts = np.load(os.path.join())
        for layer in acts.files:
            if layer not in corpus_acts:
                corpus_acts[layer] = acts[layer]
            else:
                corpus_acts[layer] = np.concatenate([corpus_acts[layer], acts[layer]])
        print(f'Doc {n_docs_consolidated}: {doc_path} ({len(toks)} tokens) --> {len(corpus_contexts)} total contexts')
        n_docs_consolidated += 1
        if n_docs_consolidated == max_docs:
            break  # Done

print(f'Found {n_docs_consolidated} docs & {len(corpus_contexts)} contexts and obtained activations of shape {corpus_acts[layer].shape}')
if max_toks:
    print(f'Ignored {n_long_docs} docs longer than {max_toks} tokens.')
out_dir_path = os.path.join(data_path, f'{n_docs_consolidated}docs_{len(corpus_contexts)}contexts_{max_toks}maxtokens')
os.mkdir(out_dir_path)
pickle.dump(corpus_contexts, open(os.path.join(out_dir_path, refs.contexts_fn), 'wb'))
np.savez(os.path.join(out_dir_path, refs.acts_fn), **corpus_acts)
