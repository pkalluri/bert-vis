import math
import numpy as np
from bokeh.palettes import Inferno, Category10, Category20, Category20c, Pastel1, Pastel2, Bokeh, Plasma, Colorblind
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, column
from bokeh.models import Div, HoverTool, ColumnDataSource, PanTool, BoxZoomTool, WheelZoomTool, ResetTool
from bokeh.palettes import Inferno, Category10, Category20, Category20c, Pastel1, Pastel2, Bokeh, Plasma
from bokeh.models.annotations import Legend, LegendItem
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


DEFAULT_PALETTE = Colorblind[8][:]
BIG_PALETTE = Plasma[256][:230]


def categorical_list_to_color_list(categorical_list, palette=BIG_PALETTE, reverse_palette=False):
    if reverse_palette:
        palette = palette[::-1]
    categories = set(categorical_list)
    n_colors_per_category = len(palette) // (len(categories) - 1)
    sampled_palette = palette[::n_colors_per_category] + (palette[-1],)
    categories_to_colors = {category: sampled_palette[i] for i, category in enumerate(categories)}
    return [categories_to_colors[elt] for elt in categorical_list]


def empty_plot(dim=None, width=250, height=250, darkmode=False):
    if dim:
        width, height = dim, dim
    p = figure(width=width, height=height)
    p.axis.visible = False
    p.grid.visible = False
    if darkmode:
        p.background_fill_color = 'grey'
    return p


def custom_hover_tool(label='hover label', border=True):
    if border:
        style = 'border-bottom-style:solid; border-width:1px;'
    else:
        style = ''
    hover_tool = HoverTool(
        tooltips = f"""<div style='{style}'>@{{{label}}}</div>"""
    )
    return hover_tool


def plot_plain(contexts, layer, acts_tag, legend=False, darkmode=True):
    """
    Given a layer, and a list of binary columns in the contexts dataframe,
    plots all activations at the specified layer, colorized to visualize all the specified columns.
    TODO: document requirements of contexts dataframe.
    """
    p = empty_plot(darkmode=darkmode)
    if legend:  # add legend
        p.height += 200
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
    p.circle('x', 'y', color='black', source=source)
    p.tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), custom_hover_tool()]
    return p


def plot_binary_columns(contexts, layer, acts_tag, color_cols, legend=False):
    """
    Given a layer, and a list of binary columns in the contexts dataframe,
    plots all activations at the specified layer, colorized to visualize all the specified columns.
    TODO: document requirements of contexts dataframe.
    """
    p = empty_plot()
    if legend:  # add legend
        p.height += 200
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
            p.circle('x', 'y', color=vis_util.DEFAULT_PALETTE[col_idx], legend_label=col, source=source)
        else:
            p.circle('x', 'y', color=vis_util.DEFAULT_PALETTE[col_idx], source=source)

    p.tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), custom_hover_tool()]
    return p


def plot_categorical_column(contexts, layer, acts_tag, color_col, legend=False):
    """
    Given a layer and a categorical column in the contexts dataframe,
    plots all activations at the specified layer, colorized to visualize the specified column.
    TODO: document requirements of contexts dataframe.
    """
    p = empty_plot()
    if legend:  # add legend
        p.height += 200
        p.add_layout(Legend(orientation='horizontal', label_text_font_size='6pt', label_width=10), 'above')

    source = ColumnDataSource(
        {
            'x': contexts[f'{layer} {acts_tag} x'],
            'y': contexts[f'{layer} {acts_tag} y'],
            'color': vis_util.categorical_list_to_color_list(contexts[color_col]),
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

    p.tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), custom_hover_tool()]
    return p


def visualize_columns(contexts, layers, acts_tag, color_cols):
    """
    Given a list of layers, creates a plot of each layer,
    colorized to depict the info in the specified columns in the contexts dataframe.
    TODO: document requirements of contexts dataframe.
    """
    if len(color_cols) == 0:
        title = Div(text=f'{acts_tag}<br>' + ' / '.join(color_cols), align='center')
        return [title] + [plot_plain(contexts, layer, acts_tag, legend=(i == 0)) for i, layer in enumerate(layers)]
    elif len(color_cols) == 1:
        title = Div(text=f'{acts_tag}<br>' + ' / '.join(color_cols), align='center')
        return [title] + [plot_categorical_column(contexts, layer, acts_tag, color_cols[0], legend=(i == 0))
                          for i, layer in enumerate(layers)]
    else:
        title = Div(text=f'{acts_tag}', align='center')
        return [title] + [plot_binary_columns(contexts, layer, acts_tag, color_cols, legend=(i == 0))
                          for i, layer in enumerate(layers)]


def custom_bokeh_tooltip(label, border=True):
    if border:
        style = 'border-bottom-style:solid; border-width:1px;'
    else:
        style = ''
    return f"""<div style='{style}'>@{{{label}}}</div>"""


# def visualize_properties(contexts, layers, acts_tag, properties):
#     """
#     Given a list of layers, creates a plot of each layer,
#     colorized to depict the info in the specified columns in the contexts dataframe.
#     TODO: document requirements of contexts dataframe.
#     """
#     if len(properties) == 0:
#         title = Div(text=f'{acts_tag}<br>' + ' / '.join(properties.keys()), align='center')
#         return [title] + [plot_plain(contexts, layer, acts_tag, legend=(i == 0)) for i, layer in enumerate(layers)]
#     elif len(properties) == 1:
#         title = Div(text=f'{acts_tag}<br>' + ' / '.join(properties.keys()), align='center')
#         return [title] + [plot_categorical_column(contexts, layer, acts_tag, properties[0], legend=(i == 0))
#                           for i, layer in enumerate(layers)]
#     else:
#         title = Div(text=f'{acts_tag}', align='center')
#         return [title] + [plot_binary_columns(contexts, layer, acts_tag, properties, legend=(i == 0))
#                           for i, layer in enumerate(layers)]
