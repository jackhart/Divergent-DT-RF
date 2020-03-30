#!/usr/bin/env python

import numpy as np
import pandas as pd
import math
from itertools import combinations
from sklearn.datasets import make_gaussian_quantiles, load_iris, load_wine


def gini(class_counts, total_count):
    if total_count == 0:
        return 1
    return 1 - np.sum((class_counts / total_count) ** 2)


def find_splits(x):
    # find the first list for all possible values of list split in two
    # TODO: This doesn't work as intended yet
    values = []
    for i in range(math.ceil(len(x) / 2) + 1):
        for temp in combinations(x[i:], i + 1):
            values.append(temp)

    return values

def create_synthetic_data_function(type_p='xor'):
    """
    create dataset function for synthetic data
    :param seed_p: int, random seed for numpy
    :param type_p: str, the dataset you want returned
    :return: function, function for creating dataset
    """
    def create_synthetic_data(hparams):
        """ create specified random dataset
        :return: (X, y, data_types)  tuple( np.array(n,p), np.array(n,), list ), synthetic data
        """
        np.random.seed(hparams.seed)
        if type_p == 'xor':
            x = np.random.uniform(low=-2, high=2, size=(hparams.n,))
            y = np.random.uniform(low=-2, high=2, size=(hparams.n,))
            c = ((x < 0) & (y < 0) | (x > 0) & (y > 0)).astype(int)
            return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1), c.reshape(-1, 1), ['n', 'n']

        if type_p == 'donut':
            x, y = make_gaussian_quantiles(cov=3.,
                                             n_samples=hparams.n, n_features=2,
                                             n_classes=2, random_state=hparams.seed)
            return np.array(x), np.array(y).reshape(-1, 1), ['n', 'n']

        if type_p == 'iris':
            x, y = load_iris(return_X_y=True)
            return np.array(x), np.array(y).reshape(-1, 1), ['n']*4

        if type_p == 'wine':
            x, y = load_wine(return_X_y=True)
            return x, y.reshape(-1, 1), ['n'] * 13

        raise NotImplementedError

    return create_synthetic_data


def load_UCI_function(type_p='votes'):
    """
    Load a tsv of movie reviews, where field 0 is review id and field -1 is review
    :param data_file_p: data file path
    :param type_p: str, the dataset you want returned
    :return: function, function for creating dataset
    """
    def load_UCI(hparams):
        if type_p == 'votes':
            votes_df = pd.read_csv(hparams.data_path, header=None)
            y = (votes_df[0] == 'republican').to_numpy().astype(int).reshape(-1, 1)

            x = votes_df[range(1, 17)].replace('n', 0).replace('y', 1)
            x = x.replace('?', 2).to_numpy()  # ? in this dataset is not missing value, but abstention
            return x, y, ['c']*16

        # TODO: create loading function for other UCI datasets
        raise NotImplementedError

    return load_UCI
