import pandas as pd
import os
import json
from nltk.tokenize import sent_tokenize

root_dir = '/john1/scr1/baom/text'
wiki_dirs = [f'{root_dir}/AA', f'{root_dir}/AB', f'{root_dir}/AC']

# search_tokens = ['Asian', 'African', 'Hispanic', 'Latino', 'Native', 'Indigenous', 'Islander', 'White', 'Caucasian']
search_tokens = ['gender', ' race', 'racial', 'trans ', 'trans-', 'immigration', 'immigrant', 'immigrate']

hits = []

num_sents = 0

for wiki_dir in wiki_dirs:
    for subdir, dirs, files in os.walk(wiki_dir):
        for f in files:
            wiki_text = os.path.join(subdir, f)
            with open(wiki_text, "r") as wiki_file:
                for article in wiki_file.readlines():
                    wiki_file = json.loads(article)
                    title = wiki_file['title']
                    text = wiki_file['text']

                    contained_tokens = []
                    for i in search_tokens:
                        if i not in text:
                            continue
                        else:
                            contained_tokens.append(i)

                    if not contained_tokens:
                        continue

                    sentences = sent_tokenize(text)
                    num_sents += len(sentences)

                    for i, sent in enumerate(sentences):
                        toks = []
                        for tok in contained_tokens:
                            if tok in sent:
                                toks.append(tok)
                        if toks:
#                             data = {"title": title, "tokens": ','.join(toks), "sentence": sent, "sentence_idx": i, "path": wiki_text}
                            data = {"title": title, "sentence": sent, "sentence_idx": i, "path": wiki_text, "toks":",".join(toks)}
                            hits.append(data)
print("Num sents: ", num_sents)
df = pd.DataFrame(hits)
df.to_csv('/john1/scr1/baom/gender_race_in_wiki.tsv', sep="\t", index=False)
