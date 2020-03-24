#!/usr/bin/env python

import numpy as np
import math
from itertools import combinations


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


def load_UCI(data_file):
    """
    Load a tsv of movie reviews, where field 0 is review id and field -1 is review
    :param data_file: data file path
    :return: X, y
    """
    # TODO: create loading function for UCI datasets

    raise NotImplementedError


def create_synthetic_data(seed=55, n=1000, type='xor'):
    """
    create specified random dataset
    :param seed: int, random seed for numpy
    :param type: str, the dataset you want returned
    :return: X, y  tuple( np.array(n,p), np.array(n,) ), synthetic data
    """
    np.random.seed(seed)
    if type == 'xor':
        X = np.random.uniform(low=-2, high=2, size=(n,)).reshape(-1, 1)
        y = np.random.uniform(low=-2, high=2, size=(n,))

        y = ( (X[:,0] < 0) & (y < 0) | (X[:,0] > 0) & (y > 0)).astype(int)
        return X, y

    raise NotImplementedError
