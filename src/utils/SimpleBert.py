from typing import List, Dict, Tuple
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import itertools


class SimpleBert:

    def __init__(self, bert_type='bert-base-cased'):
        self.bert_type = bert_type
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)
        self.model = BertModel.from_pretrained(self.bert_type,
                                               output_hidden_states=True,
                                               # Whether the model returns all hidden-states.
                                               )
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def get_contexts_and_acts(self, docs: List, tokenized=False, layers: List[str] = None) -> (List[Tuple], Dict[str, np.ndarray]):
        """Given a list of docs, tokenize and return all contexts, along with layers' hidden activations."""
        docs_contexts = []
        docs_acts = {}
        first_doc = True
        for doc in docs:
            if tokenized:
                doc = self.tokenizer.convert_tokens_to_string(
                    doc[1:-1])  # untokenize - todo: use this tokenization instead of redoing
            inputs = self.tokenizer(doc, return_tensors="pt")
            outputs = self.model(**inputs)

            # save contexts
            tokens = [self.tokenizer.decode(i).replace(' ', '') for i in inputs['input_ids'].tolist()[0]]
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


    def get_toks_and_acts(self, doc, tokenized=False) -> (List[str], Dict[str, np.ndarray]):
        """Given a doc (string or tokenized), return the tokens, along with layers' hidden activations."""
        contexts, acts = self.get_contexts_and_acts([doc], tokenized=tokenized)
        doc, _ = contexts[0]
        return doc, acts