# Visualizing BERT

## Setup
Running this code requires python<=3.7 (because it uses the tensorflow lucid library which has this requirement).
Install other requirements using the `requirements.txt` file:

`pip install -r setup/requirements.txt`

## How to use this code
####1. ACTIVATIONS
You will need to create a directory to contain your document's tokens and the activations at each layer you are interested in. Specifically, place two files in your directory:
* `tokens.pickle` - which contains the list of the document's tokens.
* `activations.npz` -  in which each numpy array contains the document's activations at a particular layer and has shape (# OF TOKENS, WIDTH OF LAYER).
See `data/intro` for an example directory.

####2. REDUCE THE ACTIVATIONS
Note - The dimensionality reduction script is not yet in the repo. TODO - add it.

To perform dimensionality reduction on the activations, run `reduce_activations.py` with your desired parameters, for example:

`python src/reduce_activations.py data/example`

This will add a new .npz file to your directory.

####3. VISUALIZE THE ACTIVATIONS
To visualize the reduced activations, run the Jupyter notebook `visualize.ipynb`, setting the parameters to point to your directory.
This notebook visualizes the evolution of your document's activations in a neat colorful way.
