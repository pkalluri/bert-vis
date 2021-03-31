import os
import json
from random import seed
from random import randint
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

num_samples = 50000
rand_range = 269


seed(1)

for i in tqdm(range(num_samples)):
    wiki_idx = randint(0, rand_range)
    alpha_idx = 'AC'
    if wiki_idx < 100:
        alpha_idx = 'AA'
    elif wiki_idx < 200:
        alpha_idx = 'AB'
    
    num_idx = str(wiki_idx % 100)
    num_idx = '0' + num_idx if len(num_idx) == 1 else num_idx

    dataset_path = f'/u/scr/nlp/wikipedia/wikiextractor/text/{alpha_idx}/'
    save_path = f'/u/scr/baom/wikipedia/{alpha_idx}/'

    wiki_file = f'{dataset_path}/wiki_{num_idx}'
    f = open(wiki_file, "r")
    f_lines = f.readlines()
    
    line_idx = randint(0, len(f_lines) - 1)
    
    wiki_path = f'{save_path}/wiki_{num_idx}'
    if not os.path.exists(wiki_path):
        os.makedirs(wiki_path)

    article = json.loads(f_lines[line_idx])
    title = (article["title"]).replace(" ", "_").replace("/", "_")
    text = article["text"]
    
    sentences = sent_tokenize(text)
    rand_sent_idx = randint(0, len(sentences) - 1)
#     print("Sentence: ", i, " ", sentences[rand_sent_idx])

    topic_path = f'{wiki_path}/{title}_{line_idx}'
    
    if not os.path.exists(topic_path):
        os.makedirs(topic_path)

    with open(f'{topic_path}/doc_sent{rand_sent_idx}.txt', 'w') as f:
        f.write(sentences[rand_sent_idx])

