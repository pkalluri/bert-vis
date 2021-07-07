"""
From a contexts file of all documents (often sentences) being considered,
this script gets the specified number of most frequent tokens
and the specified number of random tokens.
For each of these tokens, we may want to investigate how it reclusters and what complex word senses emerge at what layers.
This may serve as a random (non-cherry-picked) and rigorous investigation of small-scale/local dynamics at each layer.
"""

import pickle
import random
random.seed(0)

# Parameters
contexts_fp = 'bucket/wikipedia/1000docs_19513contexts_30maxtokens/contexts.pickle'
n_samples = 10

with open(contexts_fp, 'rb') as f:
    contexts = pickle.load(f)
all_toks = [tok for toks, tok_i in contexts for tok in toks]
tok_freq = {}
for tok in all_toks:
    if tok not in tok_freq:
        tok_freq[tok] = 1
    tok_freq[tok] += 1
unique_toks = tok_freq.keys()

# Random tokens
print(f'{n_samples} random tokens: {random.sample(unique_toks, n_samples)}')
# Most frequent tokens
frequency_ordered_toks = sorted(tok_freq.items(), key=lambda t: t[1], reverse=True)
print(f'{n_samples} most frequent tokens: {frequency_ordered_toks[:n_samples]}')
