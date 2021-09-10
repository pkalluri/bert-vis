"""Split single KNN file into a directory with pickled knn files. 
This is important when you only wish to load and analyze a few layers."""

import pickle
import click

import sys
import os
sys.path.insert(0, os.path.abspath('.'))  # add CWD to path
from src.utils import references as refs


@click.command(help='Given the path of a single knn models file, split it up into a directory of pickled knn model files.',)
@click.argument('models_filepath')
@click.option('--names', default=[f'arr_{i}' for i in range(13)], help='A name for each model (usually the layer name)')
def split_knn_file(models_filepath, names=[f'arr_{i}' for i in range(13)]):
	models_filepath = os.path.abspath(models_filepath)  # old
	models_dirpath = os.path.join(os.path.dirname(models_filepath), refs.knn_models_dirname)  # new
	os.mkdir(models_dirpath)

	with open(models_filepath, 'rb') as read_file:
		for name in names:
			print('Splitting off model', name)
			model = pickle.load(read_file)
			with open(os.path.join(models_dirpath, name)+'.pickle', 'wb') as write_file:
				pickle.dump(model, write_file)


if __name__ == '__main__':
	split_knn_file()
