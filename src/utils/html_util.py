"""Util for styling html"""


def style(s, css):
    return f'<span style="{css}">{s}</span>'


def highlight(s, color='lightpink', extra_css=''):
    return style(s, css=f'background-color: {color};{extra_css}')


def font_size(s, size):
    return style(s, f'font-size:{size}pt;')


def highlighter(color='lightpink'):
    return lambda s: highlight(s, color=color)


def bold(s):
    return f'<b>{s}</b>'


def fix_size(s, size=50):
    return style(s, f'width:{size}px; display: inline-block; text-align: center;')


def sizer(s, size):
    return fix_size(s, size=size)


def rgb_to_color(r, g, b):
    """Converts rgb values as a color string that can be used in css styles."""
    return f'rgba({r},{g},{b},1)'


def box(s, css=''):
    """Styles s as a block with a bottom border."""
    return f"<div style = 'border-bottom-style: solid; border-width: 1px; {css}'>{s}</ div >"
