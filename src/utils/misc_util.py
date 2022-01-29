def select_layers(layers, n_layers):
    step = int((len(layers)-1)/(n_layers-1)) if n_layers else 1
    return layers[::step]