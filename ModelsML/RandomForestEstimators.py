"""Implementations of Estimators for Random Forests"""

from .abstract_estimator import Estimator
from .DecisionTreeEstimators import ClassicDecisionTreeClassifier
from .util import gini, time_function
import numpy as np

class ClassicRandomForestClassifier(Estimator):
    """
    Implements Random Forest with DecisionTreeClassification objects.

    Random forest is implemented based on https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    """

    def __init__(self, data_types=None):
        """
        Default implementation
        :param data_types: list, optional, ordered list of feature types (see base Estimator class for details)
        """
        # TODO: implement init
        super().__init__()
        self.data_types = data_types
        self.tree_classifiers = []
        self.n_classes = None

    @time_function
    def train(self, x_train, y_train, hparams, data_types=None):
        """
        Default setup for training estimator
        :param x_train: np.array(shape=(n, p)),  training features
        :param y_train: np.array(shape=(n, 1)),   training classes/values
        :param data_types: list, optional,  ordered list of feature types (see base Estimator class for details)
        :param hparams: Hparams, object containing additional parameters.
                        Must Contain:
                            min_size: int, minimum number of examples in node, default is 2
                            max_depth: int, optional, maximum depth of tree
                            max_gini: int, maximum gini for split allowed, default is 1

        :return self
        """

        # TODO: implement training
        raise NotImplementedError

    @time_function
    def predict(self, x_test):
        """
        Uses averaging in the leaf nodes for classification.

        Default setup for predicting with trained estimator
        :param x_test: np.array(shape=(m, p)), testing features

        :return tuple of lists: (probabilities, predictions)
        """

        # TODO: implement predicting
        raise NotImplementedError

    def __repr__(self):
        return 'ClassicRandomForestClassifier()'


class KeRF(Estimator):
    """
    Implements KeRF with KeDTClassification objects.
    """
    # TODO: add documentation on KeRF

    def __init__(self, data_types=None):
        """
        Default implementation
        :param data_types: list, optional, ordered list of feature types (see base Estimator class for details)
        """
        # TODO: implement init
        super().__init__()
        self.data_types = data_types
        self.trees = []
        self.n_classes = None

    @time_function
    def train(self, x_train, y_train, data_types=None, min_size=2, max_depth=None, max_gini=1):
        """
        Default setup for training estimator
        :param x_train: np.array(shape=(n, p)),  training features
        :param y_train: np.array(shape=(n, 1)),   training classes/values
        :param data_types: list, optional,  ordered list of feature types (see base Estimator class for details)
        :param min_size: int, minimum number of examples in node, default is 2
        :param max_depth: int, optional, maximum depth of tree
        :param max_gini: int, maximum gini for split allowed, default is 1

        :return self
        """

        # TODO: implement training
        raise NotImplementedError

    @time_function
    def predict(self, x_test):
        """
        Uses averaging in the leaf nodes for classification.

        Default setup for predicting with trained estimator
        :param x_test: np.array(shape=(m, p)), testing features

        :return tuple of lists: (probabilities, predictions)
        """

        # TODO: implement predicting
        raise NotImplementedError

    def __repr__(self):
        return 'ClassicRandomForestClassifier()'



