from typing import List, Dict, Tuple
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import itertools


def get_contexts_and_acts(docs:List, tokenized=False, layers:List[str]=None) -> (List[Tuple], Dict[str,np.ndarray]):
    """Given a list of docs, tokenize and return all contexts, along with layers' hidden activations."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
    docs_contexts = []
    docs_acts = {}
    first_doc = True
    for doc in docs:
        if tokenized:
            doc = tokenizer.convert_tokens_to_string(doc[1:-1])  # untokenize - todo: use this tokenization instead of redoing
        inputs = tokenizer(doc, return_tensors="pt")
        outputs = model(**inputs)

        # save contexts
        tokens = [tokenizer.decode(i).replace(' ', '') for i in inputs['input_ids'].tolist()[0]]
        new_contexts = [(tokens, pos) for pos in range(len(tokens))]
        docs_contexts.extend(new_contexts)

        # save acts
        new_acts = torch.squeeze(torch.stack(outputs.hidden_states, dim=0), dim=1)
        if not layers:
        layers = [f'arr_{i}' for i in range(len(new_acts))]
        new_acts = {layer: new_acts[layer_idx].detach().numpy() for layer_idx, layer in enumerate(layers)}
        if first_doc:
            docs_acts = new_acts
            first_doc = False
        else:
            for layer in layers:
                docs_acts[layer] = np.concatenate([docs_acts[layer], new_acts[layer]])
    return docs_contexts, docs_acts


def get_toks_and_acts(doc, tokenized=False) -> (List[str], Dict[str, np.ndarray]):
    """Given a doc (string or tokenized), return the tokens, along with layers' hidden activations."""
    contexts, acts = get_contexts_and_acts([doc], tokenized=tokenized)
    doc, _ = contexts[0]
    return doc, acts

def get_masked_variants(doc: List[str], mask_lengths:List[int], verbose=False) -> List[List[str]]:
    """Returns every variant of the given doc, where a variant is the given doc
    masked with a valid number of masks (as specified in mask_lengths)."""
    masks:List[Tuple] = []  # a list of all masks (of varying length)
    for mask_len in mask_lengths:
        if verbose: print(f'getting all masks with len {mask_len}')
        assert(mask_len <= len(doc)-2)  # CLS and SEP should never be masked
        masks.extend(itertools.combinations(range(len(doc)-2), mask_len))
    if verbose: print('got masks')
    variants: List[List[str]] = []  # a list of all variant sequences
    for mask in masks:
        if verbose: print('applying mask', mask)
        variant = doc.copy()
        for tok_idx in mask:
            variant[tok_idx+1] = '[MASK]'  # skip over CLS token
        variants.append(variant)
    return variants


def mask(toks: List[str], mask_idxs: Tuple[int]):
    """Return toks, with the tokens at mask_idxs changed to '[MASK]'."""
    masked_toks = toks.copy()
    for mask_idx in mask_idxs:
        masked_toks[mask_idx] = '[MASK]'
    return masked_toks
