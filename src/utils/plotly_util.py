import plotly as px
from plotly import graph_objects as go
from typing import List
import pandas as pd
from plotly.colors import qualitative as colors
palette = colors.Plotly


def combine_figs(figs):
    fig = figs[0]
    for other_fig in figs[1:]:
        fig.add_traces(list(other_fig.select_traces()))
    return fig


def get_error_bands(keys:List[str], q1:pd.DataFrame, q3:pd.DataFrame, layers:List[str]):
    bands = go.Figure()
    for i, key in enumerate(keys):
        trace = go.Scatter(x=layers+layers[::-1], 
                           y=q1[key].tolist() + q3[key].tolist()[::-1], 
                           fill='toself',
                           fillcolor=palette[i],
                           line_color=palette[i],
                           mode='lines',
                           opacity=.2,
                          )
        trace['showlegend'] = False
        bands.add_trace(trace)
    return bands