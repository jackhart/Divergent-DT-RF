#!/usr/bin/env python

import numpy as np
import math
from itertools import combinations
from sklearn.datasets import make_gaussian_quantiles, load_iris


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

def create_synthetic_data_function(seed_p=58, n_p=1000, type_p='xor'):
    """
    create dataset function for synthetic data
    :param seed_p: int, random seed for numpy
    :param type_p: str, the dataset you want returned
    :return: function, function for creating dataset
    """
    def create_synthetic_data(seed=seed_p, n=n_p, type=type_p):
        """ create specified random dataset
        :return: (X, y, data_types)  tuple( np.array(n,p), np.array(n,), list ), synthetic data
        """
        np.random.seed(seed)
        if type == 'xor':
            x = np.random.uniform(low=-2, high=2, size=(n,))
            y = np.random.uniform(low=-2, high=2, size=(n,))

            c = ((x < 0) & (y < 0) | (x > 0) & (y > 0)).astype(int)
            return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1), c.reshape(-1, 1), ['n', 'n']

        if type == 'donut':
            x, y = make_gaussian_quantiles(cov=3.,
                                             n_samples=n, n_features=2,
                                             n_classes=2, random_state=seed)
            return np.array(x), np.array(y).reshape(-1, 1), ['n', 'n']

        if type == 'iris':
            x, y = load_iris(return_X_y=True)
            return np.array(x), np.array(y).reshape(-1, 1), ['n']*4
        raise NotImplementedError

    return create_synthetic_data


def load_UCI(data_file):
    """
    Load a tsv of movie reviews, where field 0 is review id and field -1 is review
    :param data_file: data file path
    :return: X, y
    """
    # TODO: create loading function for UCI datasets
    raise NotImplementedError