# # Probing folk wisdom

# ## Setup

import os
import pickle
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__,'../../..')))  # --> vis-wiki --> analysis-and-vis --> src
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

import utils.references as refs
from utils.acts_util import normalize, spherize
from utils.context_util import abbreviated_context as context_txt

import warnings
warnings.filterwarnings('ignore')  # temporary, not recommended


@click.command()
@click.argument('corpus_dir')
@click.argument('scores_fn')
@click.argument('corpus_filter')
@click.option('--n_layers', '-l', default=None, type=int, help='Number of layers to analyze. If not specified, use all layers.')
@click.option('--n_neighbors', '-k', default=None, type=int, help='Number of neighbors to analyze. If not specified, use default.')
@click.option('--n_points', '-n', default=10000, type=int, help='Number of points (activation and token pairs) that the models are trained on.')
@click.option('--n_classes', '-c', default=100, type=int, help='Number of classes (different tokens) that the models choose between.')
@click.option('--probe_tok', '-i', is_flag=True, help='If probe_tok, then the process is reversed: probe what the main token\'s \
                                                                embedding knows about its neighbors.')
@click.option('--concatenate', '-cc', default=None, type=int, help='If specified, concatenate each layer with the specified layer before probing.')
@click.option('--note', '-a', default='', type=str, help='Optional note to include in the appended rows.')
def probe_identity(corpus_dir:str, scores_fn:str, corpus_filter:str, n_layers:int=None, n_neighbors=3, 
    n_points=10000, n_classes=100, probe_tok=False, note:str='', concatenate:int=None):
    '''
    Given corpus activations and contexts as well as the name of a filter that will be applied (e.g. contexts with 'top-toks' only), 
    train a logistic regression to probe each activation for its own token.
    Also train logistic regressions for several of its neighbors to probe those neighboring activations for the original token.
    Update a CSV file and visualization with the scores.
    
    Args:
        corpus_dir (str): Corpus directory containing activations.npy and contexts.pickle
        scores_fn (str): Name of CSV file in corpus directory, containing scores calculated so far.
        corpus_filter (str): Name of filter to apply to corpus contexts before running identity probe, e.g. "top-toks"
        n_layers (int, optional): Number of layers to analyze. If not specified, use all layers.
        n_neighbors (int, optional): Number of neighbors to analyze. If not specified, use default.
        n_points (int, optional): Number of points (activation and token pairs) that the models are trained on.
        n_classes (int, optional): Number of classes (different tokens) that the models choose between.
        probe_tok (bool, optional): If probe_tok, then the process is reversed: 
                                    logistic regressions probe what the main token's embedding knows about its neighbors.
        note (str, optional): Optional note to include in the appended rows.
        concatenate (int, optional): If specified, concatenate each layer with specified layer before probing.
                (This is useful for studying whether certain layers contain different, possibly complementary information.)
    '''
    # include all params
    params = ' '.join(([f'{arg}={val}' for arg, val in locals().items() if arg not in ['args', 'kwargs']]))
    print(params)
    params = ' '.join([f'{arg}={val}' for arg, val in locals().items() if arg in ['n_points', 'n_classes', 'probe_tok']])
    note += params


    # load corpus and acts
    corpus_dir = os.path.abspath(corpus_dir)
    corpus_contexts = pickle.load(open(os.path.join(corpus_dir, refs.contexts_fn),'rb'))
    corpus_acts = np.load(os.path.join(corpus_dir, refs.acts_fn))
    layers = corpus_acts.files
    # get all tokens
    corpus_toks = np.array([doc[pos] for doc, pos in corpus_contexts])
    vocab = list(set(corpus_toks))

    # get filter
    corpus_filter_name = corpus_filter
    if corpus_filter_name.startswith('top'):
        if corpus_filter_name != 'top':
            n_classes = int(corpus_filter_name[len('top'):])  # e.g. if name is 'top10, override num classes
        corpus_filter = top_toks_filter(corpus_toks=corpus_toks, c=n_classes)
    elif corpus_filter_name=='random':
        corpus_filter = random_filter(corpus_toks=corpus_toks, c=n_classes)
    elif corpus_filter_name=='bert-partial':
        corpus_filter = bert_partial_tok_filter(corpus_toks=corpus_toks, c=n_classes)
    elif corpus_filter_name in ['NN', 'JJ', 'VB', 'RB']:
        corpus_filter = POS_filter(corpus_toks=corpus_toks, POS=corpus_filter_name, c=n_classes)
    print(f'Number of of instances: {len(corpus_filter)}')

    if n_points: corpus_filter = corpus_filter[:n_points]

    # calculate scores
    new_scores = _probe_identity(corpus_acts, corpus_contexts, corpus_filter, 
        n_layers=n_layers, gaps=range(-1*n_neighbors, n_neighbors+1),
        probe_tok=probe_tok, layer_to_concat=concatenate)
    new_scores['filter'] = corpus_filter_name
    new_scores['note'] = note
    # update csv
    scores_fp = os.path.join(corpus_dir, scores_fn)
    if os.path.exists(scores_fp):
        scores = pd.read_csv(scores_fp)
    else:
        scores = pd.DataFrame(columns=['layer','gap','score','filter', 'note'])
    scores = scores.append(new_scores, ignore_index=True)
    scores.to_csv(scores_fp, index=False)
    # update heatmaps
    identity_heatmaps(scores, probe_tok=probe_tok).write_html(scores_fp.split('.')[-2]+'.html')


def _probe_identity(corpus_acts, corpus_contexts, corpus_filter, n_layers:int=None,
                       gaps=range(-3,4), train_frac=.8, probe_tok=False, verbose=False,
                       layer_to_concat:int=None):
    '''
    Given activations and contexts and a filter, train a logistic regression to probe the activation for its own token.
    Also train logistic regressions for several of its neighbors to probe those neighboring activations for the original token.
    If probe_tok, then the process is reversed: logistic regressions probe what the main token's embedding knows about its neighbors.
    Return a pandas dataframe.
    '''    
    # calculate data filters
    acts_filters = {}
    contexts_filters = {}
    for gap in gaps:
        contexts_filter = []
        acts_filter = []
        for idx in corpus_filter:
            doc, pos = corpus_contexts[idx]
            if pos+gap in range(0, len(doc)):
                if probe_tok:
                    acts_filter.append(idx) # from tok
                    contexts_filter.append(idx+gap) # to my neighbors
                else:
                    acts_filter.append(idx+gap) # from neighbors
                    contexts_filter.append(idx)  # to tok
        acts_filters[gap] = acts_filter
        contexts_filters[gap] = contexts_filter
    for gap in gaps:
        if len(acts_filters[gap]) == 0:  # no contexts found
            del acts_filters[gap]
            del contexts_filters[gap]
    gaps = acts_filters.keys()
    
    # get target data
    corpus_toks = np.array([doc[pos] for doc, pos in corpus_contexts])
    target_toks = {gap: corpus_toks[contexts_filter] for gap, contexts_filter in contexts_filters.items()}
    layers = list(corpus_acts.keys())
    if layer_to_concat != None:
        print(layer_to_concat)
        layer_to_concat = layers[layer_to_concat]
        print(layer_to_concat)
        _acts_to_concat = corpus_acts[layer_to_concat]
    if not n_layers: n_layers = len(layers)

    model = LogisticRegression(max_iter=200, solver='saga', n_jobs=4)
    scaler = StandardScaler()
    scores = []
    for layer in layers[:n_layers]:
        print(layer,'...', end=' ', flush=True)
        _acts = corpus_acts[layer]
        for gap in gaps:
            print(gap,'...', end=' ', flush=True)
            _input_acts = _acts[acts_filters[gap]]
            if layer_to_concat != None:
                _input_acts = np.hstack([_input_acts, _acts_to_concat[acts_filters[gap]]])
            _input_acts = scaler.fit_transform(_input_acts)  # Standarize features
            _target_toks = target_toks[gap]
            n_train = int(train_frac * len(_input_acts))
            if verbose: print('fitting model...')
            model.fit(_input_acts[:n_train], _target_toks[:n_train])
            if verbose: print('calculating accuracy...')
            score = model.score(_input_acts[n_train:], _target_toks[n_train:])
            scores.append([f'{layer_to_concat}+{layer}',gap,score])
            if verbose: print(f'{layer}:{score}', flush=True)
        print()
    scores = pd.DataFrame(scores, columns=['layer','gap','score'])
    print(scores)
    return scores


def identity_heatmaps(scores, probe_tok=False):
    '''Given a pandas dataframe from the probing identity analysis, create a heatmap visualization'''
    title = 'Info in token' if probe_tok else 'Info about token'
    fig = px.scatter(data_frame=scores, x='gap', y='layer', size='score', color='score', size_max=10,
        color_continuous_scale=px.colors.sequential.Rainbow, title=title,
        facet_row='filter')
    # layout
    n_plots = len(scores['filter'].unique())
fig.update_layout(width=n_plots*250+100, height=350, 
                  xaxis_title='Relative position', xaxis_showticklabels=False,
                  yaxis_title='Layers', yaxis_showticklabels=False,)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))     
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
    return fig


def selected_toks_filter(corpus_toks, selected_toks):
    '''Given corpus toks and selected toks, return only the indices with a selected tok.'''
    print('Number of selected tokens: ', len(selected_toks))
    return [i for i, tok in enumerate(corpus_toks) if tok in selected_toks]


def top_toks_filter(corpus_toks, c=100):
    '''Given corpus's contexts' tokens, return only the indices that satisfy the filter.'''
    selected_toks = list(zip(*collections.Counter(corpus_toks).most_common(c)))[0]
    return selected_toks_filter(corpus_toks, selected_toks)


def random_filter(corpus_toks, c=100):
    '''Given corpus's contexts' tokens, return only the indices that satisfy the filter.'''
    selected_toks = random.sample(
        list(zip(*collections.Counter(corpus_toks).most_common(min(c*10, len(corpus_toks)))))[0], c)
    return selected_toks_filter(corpus_toks, selected_toks)


def bert_partial_tok_filter(corpus_toks, c=100):
    '''Given corpus's contexts' tokens, return only the indices that satisfy the filter.'''
    ordered_toks = list(zip(*collections.Counter(corpus_toks).most_common(len(corpus_toks))))[0]
    selected_toks = [tok for i, tok in enumerate(ordered_toks) if tok.startswith('##')][:c]
    return selected_toks_filter(corpus_toks, selected_toks)


def POS_tag(tok): 
    '''Simple method for guessing tok's part of speech tag. Doesn't use context.'''
    return nltk.pos_tag([tok,])[0][1] if tok else ''


def POS_filter(corpus_toks, POS:str='NN', c=100):
    '''Given corpus's contexts' tokens, return only the indices that satisfy the filter.'''
    ordered_toks = list(zip(*collections.Counter(corpus_toks).most_common(len(corpus_toks))))[0]
    selected_toks = []
    for tok in ordered_toks:
        if POS_tag(tok) == POS:
            selected_toks.append(tok)
        if len(selected_toks) == c:
            break
    return selected_toks_filter(corpus_toks, selected_toks)


if __name__ == '__main__':
    probe_identity()
