"""Given any data directory containing a doc.txt file, uses BERT or GPT to generate
a corresponding tokens.pickle file and a corresponding activations.npz file."""
from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel
import os
import argparse
import pickle
import torch
import numpy as np
import sys
import ast
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src import references

max_docs = None  # Max number of documents to read or None. If None, this is ignored.
max_contexts = None  # Max number of contexts to read or None. If None, this is ignored.
max_toks = 30  # Max number of tokens in an acceptable document. If None, this is ignored.

model_type = 'bert' # 'gpt'

data_path = '/john1/scr1/baom/gender_race_in_wiki.tsv'
save_path = f'/john1/scr1/baom/{model_type}/gender_race_in_wiki'

save_ext = '.tsv'

save_size = 5000 # // how many samples to save in each subdirectory

random_state = 1
frac = 1.0 # 0.02 // fraction of rows to sample from in provided .tsv file

tokenizer = None
model = None

if model_type == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
elif model_type == 'gpt':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
else:
    print("Incorrect model_type set.")
    exit()

df = pd.read_csv(data_path, sep='\t')

df_sub = df.sample(frac = frac, random_state = random_state)
df_sub = df_sub[df_sub['sentence'].map(len) < 512]

if 'tokens' in df_sub.columns:
    df_sub.drop(columns=['tokens'], inplace=True)

# Create a list of contexts. Each context will be a tuple: (doc's tokens, position in doc).
contexts = []
# Create a dictionary to map layer to list of docs' activations.
# Each doc's activations will be size (# contexts x size of embedding)
layers = {}
n_docs_consolidated = 0
n_long_docs = 0

for k in range(0, len(df_sub), save_size):

    df_sub_chunk = df_sub.iloc[k:min(k + save_size, len(df_sub) - 1)]
    df_sub_chunk = df_sub_chunk.copy()

    for _, row in tqdm(df_sub_chunk.iterrows()):

        sent = row['sentence']
        inputs = tokenizer(sent, return_tensors="pt")
        tokens = [tokenizer.decode(i).replace(' ', '') for i in inputs['input_ids'].tolist()[0]]

        try:
            outputs = model(**inputs)
            hidden_state = outputs.hidden_states
            hidden_state = torch.stack(hidden_state, dim=0)
            hidden_state = torch.squeeze(hidden_state, dim=1)
        except Exception as e:
            print(str(e))
            print(row['sentence'])
            hidden_state = ()

        for tok_i in range(len(tokens)):
            context = (tokens, tok_i)
            contexts.append(context)

        num_layers = hidden_state.shape[0]
        for l in range(num_layers):
            layer = f'arr_{l}'

            if layer not in layers:
                layers[layer] = hidden_state[l, :, :,].detach().numpy()
            else:
                layers[layer] = np.concatenate([layers[layer], hidden_state[l, :, :,].detach().numpy()])

        print(f'Doc {n_docs_consolidated}: {data_path} ({len(tokens)} tokens) --> {len(contexts)} total contexts')
        n_docs_consolidated += 1
        print(n_docs_consolidated)
        if n_docs_consolidated == max_docs:
            break  # Done

    print(f'Found {n_docs_consolidated} docs & {len(contexts)} contexts and obtained activations of shape {layers[layer].shape}')
    if max_toks:
        print(f'Ignored {n_long_docs} docs longer than {max_toks} tokens.')
    out_dir_path = os.path.join(
        save_path,
        f'{n_docs_consolidated}docs_{len(contexts)}contexts_{max_toks}maxtokens'
    )
    os.makedirs(out_dir_path)
    consolidated_contexts_fn = f'{references.CONTEXTS_BASE}.pickle'
    consolidated_acts_fn = f'{references.ACTIVATIONS_BASE}.npz'
    with open(os.path.join(out_dir_path, consolidated_contexts_fn), 'wb') as f:
        pickle.dump(contexts, f)
    np.savez(os.path.join(out_dir_path, consolidated_acts_fn), **layers)
    contexts = []
    layers = {}

