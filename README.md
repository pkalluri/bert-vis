# Visualizing BERT

## Setup
Running this code requires python<=3.7 (because it uses the tensorflow lucid library which has this requirement).
Install other requirements using the `requirements.txt` file:

`pip install -r setup/requirements.txt`

## How to use this code
You will need to create a directory to contain your document's tokens and activations. Place two files in your directory:
* `tokens.pickle` - which contains the list of the document's tokens.
* `activations.npz` -  in which each numpy array contains the document's activations at one layer and has shape (# OF TOKENS, WIDTH OF LAYER).

See `data/example` for an example directory.

To visualize the activations, simply run the Jupyter notebook `visualize.ipynb`, setting the parameters to point to your directory (and changing any other parameters you wish).
This notebook visualizes the evolution of your document's activations in a neat colorful way.
