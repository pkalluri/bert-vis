"""Given any data directory containing a doc.txt file, uses BERT to generate
a corresponding tokens.pickle file and a corresponding activations.npz file."""
from transformers import BertTokenizer, BertModel
import os
import argparse
import pickle
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
print(sys.path)
from src import references

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Path to data directory, which contains doc.txt')
args = parser.parse_args()
data_path = os.path.abspath(args.data_path)

# Get and save tokens
with open(os.path.join(data_path, f'{references.DOC_BASE}.{references.DOC_EXT}')) as f:
    text = f.read()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
inputs = tokenizer([text], return_tensors="pt")
tokens = [tokenizer.decode(i).replace(' ', '') for i in inputs['input_ids'].tolist()[0]]
with open(os.path.join(data_path, f'{references.TOKENS_BASE}.{references.TOKENS_EXT}'), 'wb') as f:
    pickle.dump(tokens, f)

# Get and save acts
model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
outputs = model(**inputs)
acts = torch.stack(outputs.hidden_states, dim=0)
acts = torch.squeeze(acts, dim=1)
n_layers = len(acts)
acts = {f'arr_{i}': acts[i].detach().numpy() for i in range(n_layers)}
acts_path = os.path.join(data_path, f'{references.ACTIVATIONS_BASE}.{references.ACTIVATIONS_EXT}')
np.savez(acts_path, **acts)
