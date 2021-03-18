import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from pathlib import Path
import os


MAX_INPUT_SIZE = 512

DIR = Path('../data/alice/sample3/')
DOC_TO_READ = DIR / 'doc.txt'

TOKENS_OUTPUT = DIR / 'tokens.pickle'
ACTIVATIONS_OUTPUT = DIR / 'activations.npz'


def tokenize_and_segment():
    file = open(DOC_TO_READ)
    lines = file.read().replace("\n", " ")
    file.close()
    
    sents = sent_tokenize(lines)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized_texts = []
    segments_ids = [0]

    for i, sent in enumerate(sents):
        marked_text = sent + " [SEP] "
        tokenized_text = tokenizer.tokenize(marked_text)
        tokenized_texts.extend(tokenized_text)
        segments_ids.extend([i] * len(tokenized_text))
        
    # Clip tokens to MAX_INPUT_SIZE
    tokenized_texts, segments_ids = tokenized_texts[:MAX_INPUT_SIZE - 1], segments_ids[:MAX_INPUT_SIZE - 1]
    tokenized_texts =  ["[CLS]"] + tokenized_texts
    segments_ids = [segments_ids[0]] + segments_ids
    
    # Pickle the tokenized text
    f = open(TOKENS_OUTPUT, 'wb')
    pkl.dump(tokenized_texts, f)
    
    tokens_tensors = []
    segments_tensors = []

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_texts)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensor


def extract_embeddings(tokens_tensor, segments_tensor):
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # token_embeddings is now of shape (13, 512, 768) aka (# layers, # tokens, width of layer)

        arr = np.array(token_embeddings)
        np.savez(ACTIVATIONS_OUTPUT, *arr)
        

tokens_tensor, segments_tensor = tokenize_and_segment()
extract_embeddings(tokens_tensor, segments_tensor)
