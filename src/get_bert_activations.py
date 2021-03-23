import torch
import numpy as np
import pickle as pkl
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from pathlib import Path
import os
import argparse
import glob


class GetEmbeddings:
    
    def __init__(self, bert_type='bert-base-cased'): 
        self.bert_type = bert_type
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)
        self.model = BertModel.from_pretrained(self.bert_type,
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        # TODO: explore custom BERT config to increase input size
        # Note that this script segments the text into subsets of 512 tokens and performs the embedding extraction on each subset
        self.MAX_INPUT_SIZE = 512 # default maximum sequence length for BERT


    def tokenize_and_extract(self, txt_file):

        file = open(txt_file)
        lines = file.read()
        file.close()

        sentences = sent_tokenize(lines)

        tokenized_texts = []
        # segment_ids are tokens used to differentiate between different sentences.
        segment_ids = []

        for i, sent in enumerate(sentences):
            marked_text = sent + " [SEP] "
            tokenized_text = self.tokenizer.tokenize(marked_text)
            tokenized_texts.extend(tokenized_text)
            segment_ids.extend([i] * len(tokenized_text))

        for i in range(0, len(tokenized_texts), self.MAX_INPUT_SIZE):
            start = i
            end = min(len(tokenized_texts), i + self.MAX_INPUT_SIZE)

            tokenized_texts_subset, segment_ids_subset = tokenized_texts[start:end - 1], segment_ids[start:end - 1]
            tokenized_texts_subset =  ["[CLS]"] + tokenized_texts_subset
            segment_ids_subset = [segment_ids_subset[0]] + segment_ids_subset
            
            data_dir = txt_file[:txt_file.rindex('/')]
            tokens_file = f'{data_dir}/tokens_{int(i / self.MAX_INPUT_SIZE)}_{self.bert_type}.pickle'

            # Pickle the tokenized text
            f = open(tokens_file, 'wb')
            pkl.dump(tokenized_texts_subset, f)

            # Map the token strings to their vocabulary indeces.
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_texts_subset)

            # Convert inputs to PyTorch tensors
            token_tensor = torch.tensor([indexed_tokens])
            segment_tensor = torch.tensor([segment_ids_subset])

            with torch.no_grad():
                outputs = self.model(token_tensor, segment_tensor)

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
                activations_file = f'{data_dir}/activations_{int(i / self.MAX_INPUT_SIZE)}_{self.bert_type}.npz'
                np.savez(activations_file, *arr)


# Run like so:  python get_bert_activations.py -d ~/bert-vis/data/wikipedia/ -b 'bert-base-cased'

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help = "directory to recursively extract embeddings for")
parser.add_argument("-b", "--bert", help = "probably want bert-base-cased or bert-large-cased")
    
# Read arguments from command line
args = parser.parse_args()

Embeddings = GetEmbeddings()

print("data dir: ", args.dir)

for txt_file in glob.iglob(args.dir + '**/*.txt', recursive=True):
    print("txt: ", txt_file)
    Embeddings.tokenize_and_extract(txt_file)

    
