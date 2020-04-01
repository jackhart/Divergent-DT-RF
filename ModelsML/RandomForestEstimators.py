"""Implementations of Estimators for Random Forests"""

from .abstract_estimator import Estimator
from .DecisionTreeEstimators import ClassicDecisionTreeClassifier
from .util import time_function
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
        self.classes = None
        self.tree_classifiers = []
        self.sample_features = []

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
                            n_trees: int, number of trees in forest
                            m_try: int, number of features to select from for tree splitting
                            bootstrap: boolean, whether to take bootstrap of sample, otherwise original sample used
                            seed: int, random seed for numpy
        :return self
        """
        np.random.seed(hparams.seed)

        n, p = x_train.shape
        assert n == y_train.shape[0], "y_train and x_train are not the same length"
        assert hparams.m_try <= p, "m_try must be less than or equal to number of x _train features"

        # set number of classes
        self.classes = np.unique(y_train)

        # set data_types to numeric if not provided
        if data_types is None:
            self.data_types = ['n'] * p
        else:
            self.data_types = data_types

        # instantiate tree classifiers
        self.tree_classifiers = [ClassicDecisionTreeClassifier() for _ in range(hparams.n_trees)]

        # subset features
        features_inx = np.array([*range(p)])
        self.sample_features = [np.random.choice(features_inx, replace=False,
                                                 size=hparams.m_try) for _ in range(hparams.n_trees)]

        # train each tree
        for tree, features in zip(self.tree_classifiers, self.sample_features):
            if hparams.bootstrap is True:
                x_subset = np.random.choice(x_train[:, features], size=n, replace=True)
            else:
                x_subset = x_train[:, features]

            # if hparams.m_try == 1:  # edge case where we are selecting 1 variable
            #   x_subset = x_subset.reshape((-1, 1))

            _, _ = tree.train(x_subset, y_train, hparams)

        return self

    @time_function
    def predict(self, x_test):
        """
        Uses averaging in the leaf nodes for classification.

        Default setup for predicting with trained estimator
        :param x_test: np.array(shape=(m, p)), testing features

        :return tuple of lists: (probabilities, predictions)
        """

        all_predictions, all_probabilities = [], []
        for tree, features in zip(self.tree_classifiers, self.sample_features):
            x_subset = x_test[:, features]
            probabilities, predictions, _ = tree.predict(x_subset)
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)

        prob_new = np.apply_along_axis(np.sum, axis=0, arr=np.array(all_probabilities)) / len(self.tree_classifiers)
        pred_new = np.apply_along_axis(self._return_prediction, axis=0, arr=np.array(all_predictions))
        return prob_new, pred_new

    @staticmethod
    def _return_prediction(predictions):
        for predction in predictions:
            values, counts = np.unique(predction, return_counts=True)
            ind = np.argmax(counts)
            return values[ind]

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



