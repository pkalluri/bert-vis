"""Given any data directory containing a doc.txt file, uses BERT to generate
a corresponding tokens.pickle file and a corresponding activations.npz file."""
from transformers import BertTokenizer, BertModel
import os
import argparse
import pickle
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src import references as refs
from src.utils import bert_util

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to data directory, which contains doc.txt, and will contain output files.')
args = parser.parse_args()
data_path = os.path.abspath(args.data_path)

# Read text
text = open(os.path.join(data_path, refs.doc_fn)).read()

# Get toks and acts
doc, doc_acts = bert_util.get_doc_toks_and_acts(text)

# Save
pickle.dump(doc, open(os.path.join(data_path, refs.toks_fn), 'wb'))
np.savez(os.path.join(data_path, refs.acts_fn), **doc_acts)
