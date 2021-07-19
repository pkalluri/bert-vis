def style(s, css=''):
    return f'<span style="{css}">{s}</span>'


def highlight(s, color='lightpink', extra_css='css'):
    return style(s, css=f'background-color: {color}; {extra_css}')


def highlighter(color='lightpink'):
    return lambda s: highlight(s, color=color)


def bold_html(s):
    return f'<b>{s}</b>'


def rgb_to_color(r, g, b):
    return f'rgba({r},{g},{b},1)'


def box(s, css=''):
    return f"<div style = 'border-bottom-style: solid; border-width: 1px; {css}'>{s}</ div >"
