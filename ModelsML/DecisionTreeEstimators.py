"""Implementations of Estimators for Decision Trees"""

from .abstract_estimator import Estimator
from .DecisionTrees import DecisionTreeClassification
from .util import gini, time_function
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
        Default implementation
        :param data_types: list, optional, ordered list of feature types (see base Estimator class for details)
        :param kwargs: keyword arguments
        """
        self.data_types = data_types

        self.tree = None
        self.simplified_tree = None   # TODO: make tree struct with simpler storage to improve classification time?
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

        :return self, ClassicDecisionTreeClassifier: a trained version of itself
        """
        n, p = x_train.shape
        assert n == y_train.shape[0], "y_train and x_train are not the same length"

        # set number of classes
        self.n_classes = np.unique(y_train)

        # set data_types to numeric if not provided
        if data_types is None:
            self.data_types = ['n'] * p

        # create tree stump and calculate gini
        _, class_distribution = np.unique(y_train, return_counts=True)

        stump_gini = gini(np.array(class_distribution), y_train.size)
        self.tree = DecisionTreeClassification(class_counts=np.array(class_distribution), n_subset=y_train.size)

        # grow tree
        self.tree.grow_tree(x_train, y_train, data_types, stump_gini, self.n_classes,
                            min_size=min_size, max_depth=max_depth, current_depth=0, max_gini=1)

        # prune tree
        # TODO: self.tree.prune()

        return self

    @time_function
    def predict(self, x_test):
        """
        Uses averaging in the leaf nodes for classification.

        Default setup for predicting with trained estimator
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
        return 'ClassicDecisionTree()'

