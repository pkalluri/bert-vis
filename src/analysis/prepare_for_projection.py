"""Given contexts and activations files, generates metadata and acativations TSV files
that are visualisable with the tensorflow embedding projector: projector.tensorflow.org"""
import pickle
import numpy as np
import os
import nltk
import re
import pathlib
import sys
parent_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, parent_path)
from src.utils import context_util

# Parameters
data_dir = 'bucket/wikipedia/1000docs_19513contexts_30maxtokens'
contexts_filename = 'contexts.pickle'
acts_filename = 'activations.npz'
only_metadata = False  # If True, will only generate metadata file and will not generate activations files.

# Load contexts and acts
with open(os.path.join(os.path.abspath(data_dir), contexts_filename), 'rb') as f:
    contexts = pickle.load(f)
acts = np.load(os.path.join(data_dir, acts_filename))

if not only_metadata:
    print('Saving activations')
    for layer in acts.files:
        print(f'Saving layer {layer}')
        np.savetxt(os.path.join(data_dir, f'{layer}.tsv'), acts[layer], delimiter='\t')

print('Saving metadata')
metadata = {}
metadata['context'] = [context_util.context_str(toks, tok_i) for toks, tok_i in contexts]
metadata['context length'] = [len(toks) for toks, tok_i in contexts]
metadata['abbreviated context'] = [context_util.abbreviated_context(toks, tok_i) for toks, tok_i in contexts]
metadata['position'] = [tok_i for toks, tok_i in contexts]
metadata['position from end'] = [len(toks)-1-tok_i for toks, tok_i in contexts]
metadata['token'] = [toks[tok_i] for toks, tok_i in contexts]
metadata['doc'] = [' '.join(toks) for toks, tok_i in contexts]
metadata['POS'] = [nltk.pos_tag(toks)[tok_i][1] for toks, tok_i in contexts]
metadata['capitalized'] = [bool(re.match('[A-Z]', tok)) for tok in metadata['token']]
metadata['partial'] = [tok.startswith('##') for tok in metadata['token']]
# Create numerical version of certain columns - for easy visualizing in embedding projector
for column in ['token', 'doc', 'POS']:
    info_to_id = dict([(info, i) for i, info in enumerate(set(metadata[column]))])
    metadata[f'{column} id'] = [info_to_id[info] for info in metadata[column]]
# Write to TSV
header = '\t'.join(metadata.keys())
rows = ['\t'.join([str(tag) for tag in row]) for row in zip(*metadata.values())]
with open(os.path.join(data_dir, f'metadata.tsv'), 'w') as f:
    f.write(f'{header}\n' + '\n'.join(rows))
