"""Util for miscellaneous tasks"""

def select_layers(layers, n_layers):
    """From list of layers, choose evenly spaced n_layers"""
    step = int((len(layers)-1)/(n_layers-1)) if n_layers else 1
    return layers[::step]

def color_str(s, color='black'):
    return "<text style=background:{}>{}</text>".format(color, s)
print_color = lambda s, color: html_print(color_str(str(s), color=color))
print_pink = lambda s: print_color(s, color='pink')
print_green = lambda s: print_color(s, color='pink')