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


    def tokenize_and_extract(self, txt_file):

        file = open(txt_file)
        lines = file.read()
        file.close()

        sentences = sent_tokenize(lines)

        for i, sent in enumerate(sentences):
            marked_text = sent + " [SEP] "
            tokenized_text = self.tokenizer.tokenize(marked_text)
            segment_id = [0] * len(tokenized_text)

            tokenized_texts_subset =  ["[CLS]"] + tokenized_text
            segment_ids_subset = [0] + segment_id
            
            data_dir = txt_file[:txt_file.rindex('/')]
            tokens_file = f'{data_dir}/tokens_sent{i}_{self.bert_type}.pickle'

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
                activations_file = f'{data_dir}/activations_sent{i}_{self.bert_type}.npz'
                np.savez(activations_file, *arr)


# Run like so:  python ~/bert-vis/src/get_bert_activations_sentence.py -d ~/bert-vis/data/wikipedia/ -b 'bert-base-cased'

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

    
