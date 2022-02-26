import os
import pickle
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import sys
sys.path.insert(0, os.path.abspath('../../..'))
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import collections
import pandas as pd
import random
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # temporary, not recommended
import kaleido

import utils.references as refs
from utils.acts_util import normalize, spherize
from utils.context_util import abbreviated_context as context_txt


def identity_heatmaps(scores, probe_tok=False, title='', colors=px.colors.sequential.Rainbow, size_max=10):
    '''Given a pandas dataframe from the probing identity analysis, create a heatmap visualization'''
    fig = px.scatter(data_frame=scores, x='gap', y='layer', size='size', color='score', size_max=size_max,
        color_continuous_scale=colors, title=title,
        facet_row='filter')
    # layout
    n_plots = len(scores['filter'].unique())
    n_gaps = len(scores['gap'].unique())
    fig.update_layout(xaxis_dtick=1, xaxis_title='Relative position', yaxis_title='', autosize=False, height=n_plots*350+100, width=n_gaps*40+100)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))     
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
    return fig