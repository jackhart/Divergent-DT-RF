"""Implementations of Estimators for Decision Trees"""

from .abstract_estimator import Estimator
from .DecisionTrees import DecisionTreeClassification, KeDTClassification
from .util import gini, time_function, entropy
import numpy as np


class ClassicDecisionTreeClassifier(Estimator):

    """
    Decision Tree Classifier implemented with CART algorithm when using default parameters.
    Only supports Classification.

    Breiman, L., Friedman, J.H., Olshen, R., and Stone, C.J., 1984. Classification and Regression Tree
    ftp://ftp.boulder.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/TREE-CART.pdf
    """

    def __init__(self, data_types=None):
        """
        Initialize estimator with default values.
        :param data_types: list, optional, ordered list of feature types (see base Estimator class for details)
        """
        super().__init__()
        self.data_types = data_types
        self.tree = None
        self.n_classes = None

    @time_function
    def train(self, x_train, y_train, hparams, data_types=None):
        """
        Train classic decision tree.

        :param x_train: np.array(shape=(n, p)),  training features
        :param y_train: np.array(shape=(n, 1)),   training classes/values
        :param data_types: list, optional,  ordered list of feature types (see base Estimator class for details)
        :param hparams: Hparams, object containing additional parameters.
                        Must Contain:
                            min_size: int, minimum number of examples in node, default is 2
                            max_depth: int, optional, maximum depth of tree
                            max_gini: int, maximum gini for split allowed, default is 1
                            metric: str, type of metric used for decision split (either entropy or gini)
                :return self, ClassicDecisionTreeClassifier: a trained version of itself
        """

        n, p = x_train.shape
        assert n == y_train.shape[0], "y_train and x_train are not the same length"

        #sort data
        indx = np.argsort(y_train, axis=0).reshape(-1)
        x_train, y_train = x_train[indx, :], y_train[indx]

        # set number of classes
        self.n_classes = np.unique(y_train)

        # set data_types to numeric if not provided
        if data_types is None:
            self.data_types = ['n'] * p
        else:
            self.data_types = data_types

        # define metric function
        if hparams.metric == "gini":
            metric_func = gini
        elif hparams.metric == "entropy":
            metric_func = entropy
        else:
            raise NotImplementedError

        # create tree stump and calculate gini
        _, class_distribution = np.unique(y_train, return_counts=True)
        stump_gini = metric_func(np.array(class_distribution), y_train.size)
        self.tree = DecisionTreeClassification(class_counts=np.array(class_distribution), n_subset=y_train.size)

        # grow tree
        self.tree.grow_tree(x_train, y_train, self.data_types, stump_gini, self.n_classes,
                            min_size=hparams.min_size, max_depth=hparams.max_depth,
                            current_depth=0, max_gini=hparams.max_gini, metric_func=metric_func)

        # prune tree
        # TODO: self.tree.prune()
        return self

    @time_function
    def predict(self, x_test):
        """
        Uses averaging in the leaf nodes for classification.
        :param x_test: np.array(shape=(m, p)), testing features

        :return tuple of lists: (probabilities, predictions)
        """
        m, p = x_test.shape

        probabilities = []
        predictions = []
        for pred_leaf in self._leaf_node(x_test):
            probabilities.append(pred_leaf.class_counts / pred_leaf.n_subset)
            predictions.append(self.n_classes[np.argmax(probabilities[-1])])

        return probabilities, predictions

    def _leaf_node(self, X):
        for x in X:
            yield self.tree.traverse(x)

    def __repr__(self):
        return 'ClassicDecisionTreeClassifier()'


class KeDTClassifier(Estimator):
    """
    Decision Tree Classifier implemented with CART algorithm when using default parameters.
    Only supports Classification. Supports KeDT.
    """

    def __init__(self, data_types=None):
        """
        Initialize estimator with default values.
        :param data_types: list, optional, ordered list of feature types (see base Estimator class for details)
        """
        super().__init__()
        self.data_types = data_types
        self.tree = None
        self.n_classes = None

    @time_function
    def train(self, x_train, y_train, hparams, data_types=None):
        """
        Train classic decision tree.

        :param x_train: np.array(shape=(n, p)),  training features
        :param y_train: np.array(shape=(n, 1)),   training classes/values
        :param data_types: list, optional,  ordered list of feature types (see base Estimator class for details)
        :param hparams: Hparams, object containing additional parameters.
                        Must Contain:
                            min_size: int, minimum number of examples in node, default is 2
                            max_depth: int, optional, maximum depth of tree
                            max_gini: int, maximum gini for split allowed, default is 1
                            metric: str, type of metric used for decision split (either entropy or gini)
                :return self, ClassicDecisionTreeClassifier: a trained version of itself
        """
        if data_types is not None:
            assert all(data_t == 'n' for data_t in data_types), "KeDTClassification currently only works for numeric data"

        n, p = x_train.shape
        assert n == y_train.shape[0], "y_train and x_train are not the same length"

        # sort data
        indx = np.argsort(y_train, axis=0).reshape(-1)
        x_train, y_train = x_train[indx, :], y_train[indx]

        # set number of classes
        self.n_classes = np.unique(y_train)

        # set data_types to numeric if not provided
        if data_types is None:
            self.data_types = ['n'] * p
        else:
            self.data_types = data_types

        # define metric function
        if hparams.metric == "gini":
            metric_func = gini
        elif hparams.metric == "entropy":
            metric_func = entropy
        else:
            raise NotImplementedError

        # create tree stump and calculate gini
        _, class_distribution = np.unique(y_train, return_counts=True)
        stump_gini = metric_func(np.array(class_distribution), y_train.size)
        self.tree = KeDTClassification(class_counts=np.array(class_distribution), n_subset=y_train.size)

        # grow tree
        self.tree.grow_tree(x_train, y_train, self.data_types, stump_gini, self.n_classes,
                            min_size=hparams.min_size, max_depth=hparams.max_depth,
                            current_depth=0, max_gini=hparams.max_gini, metric_func=metric_func)

        # prune tree
        # TODO: self.tree.prune()
        return self

    @time_function
    def predict(self, x_test):
        """
        Uses averaging in the leaf nodes for classification.
        :param x_test: np.array(shape=(m, p)), testing features

        :return tuple of lists: (probabilities, predictions)
        """
        m, p = x_test.shape

        probabilities = []
        predictions = []
        for x in x_test:
            final_node = self.tree.traverse(x)
            probabilities.append(final_node.class_counts / final_node.n_subset)
            predictions.append(self.n_classes[np.argmax(probabilities[-1])])

        return (probabilities, predictions)

    def __repr__(self):
        return 'KeDTClassifier()'


