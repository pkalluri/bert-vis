"""Util for working with colors and bokeh plots."""
import math
import numpy as np
from typing import List
import pandas as pd
from bokeh.palettes import Inferno, Category10, Category20, Category20c, Pastel1, Pastel2, Bokeh, Plasma, Colorblind
DEFAULT_PALETTE = Colorblind[8][:]
BIG_PALETTE = Plasma[256][:230]
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, column
from bokeh.models import Div, HoverTool, ColumnDataSource, PanTool, BoxZoomTool, WheelZoomTool, ResetTool
from bokeh.models.annotations import Legend, LegendItem
from bokeh.io import output_notebook
import math


def hue_to_rgb(ang, warp=True):
    """
    Given a hue of a particular angle, produces a corresponding RGB unit vector.
    Code adapted from Tensorflow Lucid library.
    """
    ang = ang - 360 * (ang // 360)
    colors = np.asarray([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
    ])
    colors = colors / np.linalg.norm(colors, axis=1, keepdims=True)
    R = 360 / len(colors)
    n = math.floor(ang / R)
    D = (ang - n * R) / R

    if warp:
        # warping the angle away from the primary colors (RGB)
        # helps make equally-spaced angles more visually distinguishable
        adj = lambda x: math.sin(x * math.pi / 2)
        if n % 2 == 0:
            D = adj(D)
        else:
            D = 1 - adj(1 - D)

    v = (1 - D) * colors[n] + D * colors[(n + 1) % len(colors)]
    return v / np.linalg.norm(v)


def channels_to_rgbs(X):
    """
    Given n-dimensional vectors, converts to representative 3-dimensional RGB values.
    In particular, the n dimensions are mapped to n colors,
    and each n-dimensional vector is converted to the corresponding linear combination of the n colors.
    Code adapted from Tensorflow Lucid library.
    """
    if (X < 0).any():
        X = np.concatenate([np.maximum(0, X), np.maximum(0, -X)], axis=-1)

    K = X.shape[-1]
    rgbs = 0
    for i in range(K):
        ang = 360 * i / K  # angle of the ith point
        color = hue_to_rgb(ang)  # color of the ith point
        color = color[tuple(None for _ in range(1))]  # reshape if necessary
        # convert the ith column (list of values) into a list of vectors all in the chosen direction
        rgbs += X[..., i, None] * color
    # We now have 3D vectors, but we need to normalize them into RGBs
    rgbs /= 1e-4 + np.linalg.norm(rgbs, axis=-1, keepdims=True)  # normalize so each 3-dim vector is unit length
    rgbs *= 255
    return rgbs.astype(int)


def darken(r,g,b, val):
    """Make rgb, val shades darker."""
    return max(r-val, 0), max(g-val, 0), max(b-val, 0)


def categorical_list_to_color_list(categorical_list, palette=BIG_PALETTE, reverse_palette=False):
    """Turn a categorical list into an equal length color list,
    so each category has a corresponding color."""
    if reverse_palette:
        palette = palette[::-1]
    categories = set(categorical_list)
    n_colors_per_category = len(palette) // (len(categories))
    sampled_palette = palette[::n_colors_per_category] + (palette[-1],)
    categories_to_colors = {category: sampled_palette[i] for i, category in enumerate(categories)}
    return [categories_to_colors[elt] for elt in categorical_list]


def empty_plot(size=None, width=250, height=250, darkmode=False):
    """Create a simple plot."""
    if size:
        width, height = size, size
    p = figure(width=width, height=height)
    p.axis.visible = False
    p.grid.visible = False
    p.border_fill_color = 'grey'
    if darkmode:
        p.background_fill_color = 'grey'
    return p


def hover_tool(label='hover label', border=True):
    """Create a bokeh hover tool, such that upon hover
    the part of source with the given label will be shown."""
    if border:
        style = 'border-bottom-style:solid; border-width:1px;'
    else:
        style = ''
    tool = HoverTool(
        tooltips=f"""<div style='{style}'>@{{{label}}}</div>"""
    )
    return tool


def plot_plain(contexts, layer, acts_tag, darkmode=True, size=300):
    """
    Given a layer, and a column in the contexts dataframe,
    plots all points at the specified layer.
    TODO: document requirements of contexts dataframe.
    """
    p = empty_plot(darkmode=darkmode, size=size)
    source = ColumnDataSource(
        {
            'x': contexts[f'{layer} {acts_tag} x'],
            'y': contexts[f'{layer} {acts_tag} y'],
            'token': contexts['token'],
            'abbreviated context': contexts['abbreviated context'],
            'abbreviated context html': contexts['abbreviated context html'],
            'context html': contexts['context html']
        }
    )
    p.circle('x', 'y', color='black', source=source)
    p.tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), hover_tool('context html')]
    return p


def plot_binary_columns(contexts, layer, acts_tag, color_cols, legend=False, size=300):
    """
    Given a layer, and a list of binary columns in the contexts dataframe,
    plots all activations at the specified layer, colorized to visualize all the specified columns.
    TODO: document requirements of contexts dataframe.
    """
    p = empty_plot(size=size)
    if legend:  # add legend
        p.height += 50
        p.add_layout(Legend(orientation='vertical', label_text_font_size='6pt', label_width=10), 'above')

    # add all contexts in grey
    source = ColumnDataSource(
        {
            'x': contexts[f'{layer} {acts_tag} x'],
            'y': contexts[f'{layer} {acts_tag} y'],
            'token': contexts['token'],
            'abbreviated context': contexts['abbreviated context'],
            'abbreviated context html': contexts['abbreviated context html'],
            'context html': contexts['context html']
        }
    )
    p.circle('x', 'y', color='lightgrey', source=source)
    # add each column's contexts in a color
    for col_idx, col in enumerate(color_cols):
        selected_contexts = contexts[contexts[col]]
        source = ColumnDataSource(
            {
                'x': selected_contexts[f'{layer} {acts_tag} x'],
                'y': selected_contexts[f'{layer} {acts_tag} y'],
                'token': selected_contexts['token'],
                'abbreviated context': selected_contexts['abbreviated context'],
                'abbreviated context html': selected_contexts['abbreviated context html'],
                'context html': selected_contexts['context html'],
            }
        )
        if legend:
            p.circle('x', 'y', color=DEFAULT_PALETTE[col_idx], legend_label=col, source=source)
        else:
            p.circle('x', 'y', color=DEFAULT_PALETTE[col_idx], source=source)

    p.tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), hover_tool('context html')]
    return p


def plot_categorical_column(contexts, layer, acts_tag, color_col, legend=False, size=300):
    """
    Given a layer and a categorical column in the contexts dataframe,
    plots all activations at the specified layer, colorized to visualize the specified column.
    TODO: document requirements of contexts dataframe.
    """
    p = empty_plot(size=size)
    # if legend:  # add legend
        # p.height += 10
        # p.add_layout(Legend(orientation='horizontal', label_text_font_size='6pt', label_width=10), 'above')

    source = ColumnDataSource(
        {
            'x': contexts[f'{layer} {acts_tag} x'],
            'y': contexts[f'{layer} {acts_tag} y'],
            'color': categorical_list_to_color_list(contexts[color_col]),
            'legend label': contexts[color_col],
            'token': contexts['token'],
            'abbreviated context': contexts['abbreviated context'],
            'abbreviated context html': contexts['abbreviated context html'],
            'context html': contexts['context html']
        }
    )
    if legend:
        p.circle('x', 'y', color='color', legend_group='legend label', source=source)
    else:
        p.circle('x', 'y', color='color', source=source)

    p.tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), hover_tool('context html')]
    return p


def visualize_columns(contexts:pd.DataFrame, layers, acts_tag:str, color_cols:List[str], size=300):
    """
    Given a contexts dataframe of layers' points and properties, creates a list
    containing a plot of each layers' points, with points colorized to depict the info
    in the specified columns in the contexts dataframe.
    TODO: document requirements of contexts dataframe.
    """
    if len(color_cols) == 0:
        title = Div(text=f'{acts_tag}<br>' + ' / '.join(color_cols), align='center')
        return [title] + [plot_plain(contexts, layer, acts_tag, size=300) for i, layer in enumerate(layers)]
    elif len(color_cols) == 1:
        title = Div(text=f'{acts_tag}<br>' + ' / '.join(color_cols), align='center')
        return [title] + [plot_categorical_column(contexts, layer, acts_tag, color_cols[0], legend=(i == 0), size=size)
                          for i, layer in enumerate(layers)]
    else:
        title = Div(text=f'{acts_tag}', align='center')
        return [title] + [plot_binary_columns(contexts, layer, acts_tag, color_cols, legend=(i == 0), size=size)
                          for i, layer in enumerate(layers)]


def title_cell(name):
    return Div(text=name, align=('center', 'center'))


class Grid:
    """Grid of bokeh elements or plots."""
    def __init__(self, row_names=None):
        self.columns = []
        if row_names:
            header_column = [None] + [title_cell(name) for name in row_names]
            self.columns.append(header_column)
        self.plots = None

    def add_column(self, name, plots):
        column = [title_cell(name)] + plots
        self.columns.append(column)

    def show(self):
        show(gridplot(zip(*self.columns)))


def plot_evolution(labels: List[str], points: List[List], distance_mats=None, colors='limegreen', line_color='black'):
    n_stages = len(points)
    n_points = len(points[0])
    if type(colors) is str:
        print('Expand')
        colors = [[colors] * n_points] * n_stages

    plots = []
    for stage_idx in range(n_stages):
        xs, ys = zip(*points[stage_idx])
        stage_colors = colors[stage_idx]
        info = ColumnDataSource({'x': xs, 'y': ys,
                                 'color': stage_colors,
                                 'label': labels})
        p = empty_plot(size=200, darkmode=False)
        p.circle(x='x', y='y', color='color', source=info)

        if distance_mats:
            distance_matrix = distance_mats[stage_idx]
            endpoints = [(xs[row_idx], ys[row_idx], xs[col_idx], ys[col_idx], distance)
                         for row_idx, row in enumerate(distance_matrix) for col_idx, distance in enumerate(row)]
            xs0, ys0, xs1, ys1, distances = zip(*endpoints)
            info = ColumnDataSource({'x0': xs0, 'y0': ys0,
                                     'x1': xs1, 'y1': ys1,
                                     'color': [line_color] * n_points**2,
                                     'label': distances})
            p.segment(x0='x0', y0='y0', x1='x1', y1='y1', line_color='color', source=info)

        p.tools = [WheelZoomTool(), PanTool(), hover_tool('label')]
        plots.append(p)
    return plots
