""" Estimator API"""
from abc import ABCMeta, abstractmethod


class Estimator(metaclass=ABCMeta):
    """Abstract class for an Estimator"""

    def __init__(self, *args, **kwargs):
        """
        Default implementation
        :param args: positional arguments
        :param kwargs: keyword arguments
        """

    @abstractmethod
    def train(self, x_train, y_train, data_types=None, *args):
        """
        Default setup for training estimator
        :param x_train: np.array(shape=(n, p)), training features
        :param y_train: np.array(shape=(n, )),  training classes/values
        :param data_types: list, optional,  ordered list of size p containing feature types
                            `n`: numeric
                            `o`: ordinal categorical
                            `c`: categorical, no order
                            If None, assumes all values are numeric.
        """
        return NotImplementedError

    @abstractmethod
    def predict(self, x_test, y_test, *args):
        """
        Default setup for predicting with trained estimator
        :param x_test: np.array(shape=(m, p)), testing features
        :param y_test: np.array(shape=(m, )),  testing classes/values
        """
        return NotImplementedError

    def __str__(self):
        return self.__repr__()