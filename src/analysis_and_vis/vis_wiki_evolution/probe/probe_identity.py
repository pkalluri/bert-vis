'''Probing folk wisdom'''
import os
import sys

sys.path.insert(0, os.path.abspath('src'))
import pickle
import numpy as np

np.set_printoptions(precision=3, suppress=True)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import collections
import pandas as pd
import random

random.seed(0)
import nltk
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import click
import inspect
from typing import List, Callable
from collections.abc import Iterable

import utils.references as refs
from utils.acts_util import normalize, spherize
from utils.context_util import get_tok, get_pos, get_doc
from utils.context_util import abbreviated_context as context_txt
from utils.context_attributes import *


@click.command()
@click.argument('corpus_dir')
@click.argument('corpus_filter')
@click.option('--scores_filename', '-o',
              help='Path to CSV file to save scores. Can be an existing CSV file to append to.',
              default='probes/probe_identity.csv', type=str, show_default=True)
@click.option('--n_layers', '-l',
              help='Number of layers to analyze. If not specified, use all layers.',
              default=None, type=int)
@click.option('--n_neighbors', '-k',
              help='Number of neighbors to analyze. If not specified, use default.',
              default=5, type=int, show_default=True)
@click.option('--n_points', '-n',
              help='Max number of points (activation and token pairs) that the probes are trained on.',
              default=5000, type=int, show_default=True)
@click.option('--n_types', '-c',
              help='Number of classes (different types) that the probes choose between.',
              default=100, type=int, show_default=True)
@click.option('--concatenate', '-cc',
              help='If specified, concatenate each layer with the specified layer before probing.',
              default=None, type=int)
@click.option('--note', '-a',
              help='Optional note to include in the appended rows.',
              default='', type=str)
@click.option('--threshold', '-t',
              help='Minimum number of frequency in order to consider this type. Default is 10.',
              default=100, type=int)
def token_probe(corpus_dir:str, corpus_filter:str, scores_filename:str,
                n_layers:int, concatenate:int, n_neighbors:int,
                n_points:int, n_types:int, note:str, threshold:int, max_iter: int = 100):
    """
    Given corpus activations and contexts as well as the name of a filter that will be applied (e.g. contexts with
    'top-toks' only), train a logistic regression to probe each activation for its own token. Also train logistic
    regressions for several of its neighbors to probe those neighboring activations for the original token. Update a
    CSV file and visualization with the scores.

    Args:

        corpus_dir (str): Path to corpus directory containing activations.npz and contexts.pickle

        corpus_filter (str): Name of filter to apply to corpus contexts before running identity probe.
        One of: top, random, bert-partial, gpt-partial

    """
    # Print params
    params = '\n'.join(([f'{arg}={val}' for arg, val in locals().items() if arg not in ['args', 'kwargs']]))
    print(f'Parameters:\n{params}\n')
    # Jot down for later
    params = ' '.join([f'{arg}={val}' for arg, val in locals().items() if arg in ['n_types', 'probe_tok']])
    note += params
    attribute = corpus_filter

    print('Loading corpus and acts...')
    corpus_dir = os.path.abspath(corpus_dir)
    corpus_contexts = pickle.load(open(os.path.join(corpus_dir, refs.toks_fn), 'rb'))
    corpus_acts = np.load(os.path.join(corpus_dir, refs.acts_fn))
    layers = corpus_acts.files

    print('Filter for tokens of interest...')
    if attribute == 'top':
        corpus_toks = np.array([get_tok(*context) for context in corpus_contexts])
        top_toks = list(zip(*collections.Counter(corpus_toks).most_common(n_types)))[0]
    filt = {'top': lambda context: get_tok(*context) in top_toks,
            'random': lambda context: True,
            'gpt-partial': is_gpt_partial,
            'bert-partial': is_bert_partial}[attribute]
    # filter for all toks that satisfy attribute and have required neighbors
    neighbors = range(-1 * n_neighbors, n_neighbors + 1)
    valid_ids_and_toks = []
    corpus_contexts_shuffled = random.sample(list(enumerate(corpus_contexts)), len(corpus_contexts))
    for (id_, tok) in corpus_contexts_shuffled:
        if ((filt(tok)) and
                (get_pos(*tok) + min(neighbors) in range(0, len(get_doc(*tok)))) and
                (get_pos(*tok) + max(neighbors) in range(0, len(get_doc(*tok))))):
            valid_ids_and_toks.append((id_, get_tok(*tok)))
    # choose sufficiently common types in this set of tokens
    valid_types = [tok for tok, count in
                   collections.Counter(list(zip(*valid_ids_and_toks))[1]).items()
                   if count > threshold]
    target_types = random.sample(valid_types, min(n_types, len(valid_types)))
    print(f'\t(Targets: {target_types})')
    # get sufficient tokens of each type
    target_ids = {type_: [] for type_ in target_types}
    for id_, tok in valid_ids_and_toks:
        for type in target_types:
            if len(target_ids[type]) < n_points:
                target_ids[type].append(id_)
        if all([len(ids) == n_points for type_, ids in target_ids.items()]):
            break  # found all tokens needed, stop looking
    target_ids = [id_ for type_, ids in target_ids.items() for id_ in ids]  # flatten

    print('Training probes...')
    scores = _tok_probe(corpus_acts, corpus_contexts, target_ids,
                        n_layers=n_layers, neighbors=range(-1 * n_neighbors, n_neighbors + 1),
                        layer_to_concat=concatenate)
    # update csv and heatmap
    scores['filter'] = attribute
    scores['note'] = note
    scores_fp = os.path.join(corpus_dir, scores_filename)
    scores_filedir, scores_filename = os.path.split(scores_fp)
    if not os.path.exists(scores_filedir):
        os.mkdirs(scores_filedir)
    scores_table = pd.read_csv(scores_fp) if os.path.exists(scores_fp) else pd.DataFrame()
    scores_table = scores_table.append(scores, ignore_index=True)
    scores_table.to_csv(scores_fp, index=False)
    identity_heatmaps(scores_table).write_html(scores_fp.split('.')[-2] + '.html')


def _tok_probe(corpus_acts, corpus_contexts,
               target_ids: List[int],
               n_layers: int = None,
               neighbors=[-3, -2, -1, 0, 1, 2, 3],
               train_frac: float = .8,
               verbose: bool = False,
               layer_to_concat: int = None,
               max_iter: int = 100):
    """
    Train a logistic regression to probe the activation for its own token. Also train logistic regressions for
    several of its neighbors to probe those neighboring activations for the original token. Return a pandas dataframe.
    """
    n_train = int(train_frac * len(target_ids))
    print(f'\t(Number of training points: {n_train})')

    # prep targets
    target_ids = np.array(target_ids)
    target_toks = [get_tok(*corpus_contexts[id_]) for id_ in target_ids]

    # probe
    model = LogisticRegression(max_iter=max_iter, solver='saga', n_jobs=4)
    scaler = StandardScaler()
    scores = []
    layers = list(corpus_acts.keys())
    if not n_layers: n_layers = len(layers)
    if layer_to_concat != None:
        _extra_acts = corpus_acts[layers[layer_to_concat]]
    for layer in layers[:n_layers]:
        layer_name = layer.split('arr_')[-1]
        if layer_to_concat: layer_name += f'+{layer_to_concat.split("arr_")[-1]}'
        print(layer_name, '...', end=' ', flush=True)
        _acts = corpus_acts[layer]
        for neigh in neighbors:
            if verbose: print(neigh)
            _input_acts = _acts[target_ids + neigh]
            if layer_to_concat != None:
                _input_acts = np.hstack([_input_acts,
                                         _extra_acts[target_ids + neigh]])
            _input_acts = scaler.fit_transform(_input_acts)  # standarize 
            if verbose: print('fit...')
            model.fit(_input_acts[:n_train], target_toks[:n_train])
            if verbose: print('evaluate...')
            score = model.score(_input_acts[n_train:], target_toks[n_train:])
            probs = model.predict_proba(_input_acts[n_train:])
            target_idxes = [list(model.classes_).index(tok) for tok in target_toks[n_train:]]
            target_probs = probs[np.arange(len(probs)), target_idxes].mean()
            scores.append([layer_name, neigh, score, target_probs])
            if verbose: print(scores)
    scores = pd.DataFrame(scores, columns=['layer', 'neighbor', 'score', 'target_probs'])
    if verbose: print(scores)
    return scores


def identity_heatmaps(scores, width=300, height=350):
    """
    Given a pandas dataframe from the probing identity analysis,
    create a heatmap visualization.
    """
    title = 'Info about token'
    fig = px.scatter(
        data_frame=scores, x='neighbor', y='layer', size='score', size_max=10,
        color='target_probs', facet_row='filter', title=title,
        color_continuous_scale=px.colors.sequential.Rainbow)
    # layout
    n_plots = len(scores['filter'].unique())
    n_neighbors = len(scores['neighbor'].unique())
    fig.update_layout(width=n_neighbors * 50 + 100, height=350 * n_plots + 100,
                      xaxis_title='Relative position', xaxis_showticklabels=False,
                      yaxis_title='Layers', yaxis_showticklabels=False, )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
    return fig


if __name__ == '__main__':
    token_probe()
