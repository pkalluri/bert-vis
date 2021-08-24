from typing import List
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import torch


def get_doc_acts(doc: str) -> (List[str], List[np.ndarray]):
    """Given a doc, tokenize and return tokens and all layers' hidden activations."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer(doc, return_tensors='pt')
    doc_toks = [tokenizer.decode(i).replace(' ', '') for i in inputs['input_ids'].tolist()[0]]
    model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True)
    outputs = model(**inputs)
    print(model.generate(inputs))
    doc_acts = torch.squeeze(torch.stack(outputs.hidden_states, dim=0), dim=1)
    doc_acts = [_doc_acts.detach().numpy() for _doc_acts in doc_acts]
    return doc_toks, doc_acts
