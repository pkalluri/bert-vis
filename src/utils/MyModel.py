"""General form of BERT/GPT/etc model, with convenient functions for getting tokens' activations."""

from typing import List, Dict, Tuple
import numpy as np
from transformers import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
import torch
import itertools
from .Token import Token
from .Activations import Activations
from utils.ModelType import ModelType, get_generic, berts, gpts


class MyModel:

    def __init__(self, model_type=ModelType.bert_base_cased):
        self.model_type = model_type
        if model_type in berts:
            self.tokenizer = BertTokenizer.from_pretrained(model_type.value)
            self.model = BertModel.from_pretrained(model_type.value, output_hidden_states=True)
        elif model_type in gpts:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_type.value)
            self.model = GPT2Model.from_pretrained(model_type.value, output_hidden_states=True)
        else:
            raise ValueError('This model is not a bert or gpt.')
        self.model.eval()


    def get_docs_acts(self, docs: List[str], tokenized=False, layers: List[str] = None) -> (List[Token], Dict[str, Activations]):
        """Given a list of docs, tokenize and return all toks, along with layers' hidden activations."""
        toks = []
        acts = {}
        first_doc = True
        for doc in docs:
            if self.model_type in berts:
                if tokenized:
                    doc = self.tokenizer.convert_tokens_to_string(
                    doc[1:-1])  # untokenize - todo: use this tokenization instead of redoing
                inputs = self.tokenizer(doc, return_tensors="pt")
            elif self.model_type in gpts:
                if tokenized:
#                     encoded_chunks = [self.tokenizer.encode(type_) for type_ in doc]
#                     ids = [id_ for chunk in encoded_chunks for id_ in chunk]
                    ids = self.tokenizer.convert_tokens_to_ids(doc)
                    inputs = {'input_ids': torch.Tensor([ids]).to(torch.long),
                                'attention_mask': torch.tensor([[1]*len(ids)])}
                else:
                    inputs = self.tokenizer(doc, return_tensors="pt")
            outputs = self.model(**inputs)
            # save new_types
            new_types = [self.tokenizer.decode(i) for i in inputs['input_ids'].tolist()[0]]
            new_types = [type_.replace(' ','') for type_ in new_types]
            new_toks = [Token(new_types, pos, self.model_type) for pos in range(len(new_types))]
            toks.extend(new_toks)

            # save acts
            new_acts = torch.squeeze(torch.stack(outputs.hidden_states, dim=0), dim=1)
            if not layers:
                layers = [f'arr_{i}' for i in range(len(new_acts))]
            new_acts = {layer: new_acts[layer_idx].detach().numpy() for layer_idx, layer in enumerate(layers)}
            if first_doc:
                acts = new_acts
                first_doc = False
            else:
                for layer in layers:
                    acts[layer] = np.concatenate([acts[layer], new_acts[layer]])
        return toks, acts


    def get_doc_acts(self, doc:str) -> (List[str], Dict[str, Activations]):
        """Given a doc, tokenize and return layers' hidden activations."""
        toks, acts = self.get_docs_acts([doc])
        return toks[0].doc, acts


    def get_toks_acts(self, toks:List[Token]) -> Dict[str, Activations]:
        """Given a list of tokens, return layers' hidden activations."""
        acts = {}
        for tok in toks:
            _, doc_acts = self.get_docs_acts([tok.doc], tokenized=True)
            for layer in doc_acts:
                if layer not in acts:
                    acts[layer] = doc_acts[layer][tok.pos]  # todo - reshape
                else:
                    acts[layer] = np.vstack([acts[layer], doc_acts[layer][tok.pos]])
        return acts


    def get_tok_acts(self, tok:Token) -> Dict[str, Activations]:
        """Given one token, return layers' hidden activations."""
        return self.get_toks_acts([tok,])
