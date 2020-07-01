"""Implementations of Decision Trees"""

from .util import find_splits, gini
from scipy.stats import gaussian_kde
import numpy as np

class GeneralDecisionTree(object):
    """
    General Decision Tree implemented to support a CART algorithm
    Breiman, L., Friedman, J.H., Olshen, R., and Stone, C.J., 1984. Classification and Regression Tree
    ftp://ftp.boulder.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/TREE-CART.pdf
    """

    def __init__(self, name='root', children=None, split_rule=None, split_feature=None):
        self.name = name
        self.split_rule = split_rule  # e.g. lambda x: 0 if ... else 1 (returns indices of children)
        self.split_feature = split_feature

        self._children = []
        if children is not None:
            self.children = children

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        assert len(children) == 2, "DecisionTree is a binary tree, must add two children at a time"

        for child in children:
            assert isinstance(child, GeneralDecisionTree), "Children must be DecisionTree objects"
            self._children.append(child)

    @children.deleter
    def children(self):
        for child in self._children:
            self._children.remove(child)

    def traverse(self, x):
        """Method for traversing tree given some example from data to a leaf node"""
        node = self
        while node.children:
            try:
                child_index = node.split_rule(x[node.split_feature])
            except TypeError:
                return node  # edge case if test data contains NA
            node = node.children[np.asscalar(child_index)]

        return node

    def __repr__(self):
        return f"Tree(name='{self.name}', children={[child for child in self.children]})"

    def __str__(self):
        return self.__repr__()


class DecisionTreeClassification(GeneralDecisionTree):
    """
    Decision Tree implemented to support CART algorithm for Classification by saving only class counts
    Breiman, L., Friedman, J.H., Olshen, R., and Stone, C.J., 1984. Classification and Regression Tree
    ftp://ftp.boulder.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/TREE-CART.pdf
    """
    # TODO: Does not yet handle missing data exactly like algorithm described by Breiman

    def __init__(self, class_counts, n_subset, name='root',
                 children=None, split_rule=None, split_feature=None):
        """
        Initialize decision tree with functionality specific for traditional classification.
        :param class_counts, np.array, counts of examples for each class in node
        :param n_subset, int, number of examples represented by this node
        :param name, str, (default='root'), see parent class
        :param children, list, (default=None), see parent class
        :param split_rule, lambda function, (default=None), function that returns index of child for split
        :param split_feature, int, (default=None), index of feature to split on in dataset for current node
        """

        super().__init__(name=name, children=children,
                         split_rule=split_rule, split_feature=split_feature)

        self.class_counts = class_counts
        self.n_subset = n_subset

    def grow_tree(self, X, y, data_types, best_gini, classes, metric_func=gini,
                  min_size=2, max_depth=None, current_depth=0, max_gini=1):
        """
        Grows tree for given dataset for a classification task.  Recursive.
        :param X, np.array, subset of variables to split on
        :param y, np.array, subset of classes
        :param data_types, list, data types (i.e. numeric, ordinal, or categorical) of X
        :param best_gini, double, gini of the current node being split
        :param classes, np.array, array of all possible class values
        :param metric_func, function,  (default = "gini"), type of splitting used, either "gini" or "entropy"
        :param min_size, int, (default=2) minimum allowable number of examples making up a node
        :param max_depth, int, (default=None) maximum number of branches off nodes allowed
        :param current_depth, used in recursion to keep track of tree depth
        :param max_gini, int, (default=1) maximum gini you allow for a split to happen
        """
        if (y.size < min_size) or (best_gini == 0.0):
            # stopping criterion: node be smaller than min size
            # if node is pure, don't split
            return
        if max_depth is not None:
            if current_depth > max_depth:
                # stopping criterion: cannot build tree greater than max size
                return

        best_thr, best_p_ind, best_type = None, None, None

        for idx, data_type in zip(range(X.shape[1]), data_types):
            x = X[:, idx]

            new_thr, new_gini, left_distribution, right_distribution = self._best_split_classification(x, y, data_type,
                                                                                                       classes, metric_func=metric_func)

            if new_gini < best_gini:  # minimize gini
                best_gini, best_thr, best_p_ind, best_type, best_left_dist, best_right_dist = new_gini, new_thr, idx, data_type, left_distribution, right_distribution

        # if better split found
        if (best_thr is not None) and (max_gini > best_gini):

            # set current tree values for split
            self.split_feature = best_p_ind
            if best_type in ['n', 'o']:
                self.split_rule = lambda x_val: (x_val > best_thr).astype(int)
            else:
                self.split_rule = lambda x_val: (x_val == best_thr).astype(int)

            # calculate class distributions for children

            splits = self.split_rule(X[:, best_p_ind])

            # subset data for splits
            right_y, right_x = y[splits == 1],  X[splits == 1, :]
            left_y, left_x = y[splits == 0], X[splits == 0, :]

            if (right_y.size < min_size) or (left_y.size < min_size):
                # stopping criterion: if either child is less than min size, don't split
                self.split_rule = None
                self.split_feature = None
                return

            # grow left child
            left_tree = DecisionTreeClassification(name=f"{self.name}_{best_p_ind}_child1",
                                                   class_counts=np.array(best_left_dist),
                                                   n_subset=np.sum(best_left_dist))

            left_tree.grow_tree(left_x, left_y, data_types, metric_func(np.array(best_left_dist), left_y.size),
                                classes=classes, min_size=min_size, max_depth=max_depth,
                                current_depth=current_depth + 1, metric_func=metric_func)

            # grow right child
            right_tree = DecisionTreeClassification(name=f"{self.name}_{best_p_ind}_child2",
                                                    class_counts=np.array(best_right_dist),
                                                    n_subset=np.sum(best_right_dist))

            right_tree.grow_tree(right_x, right_y, data_types, metric_func(np.array(best_right_dist), right_y.size),
                                 classes=classes, min_size=min_size, max_depth=max_depth,
                                 current_depth=current_depth + 1, metric_func=metric_func)

            # add children to tree
            self.children = [left_tree, right_tree]

        else:
            # stopping criterion: gini not improved
            # gini is greater than user-specified maximum gini
            # gini is greater than user-specified maximum gini
            return

    def prune(self):
        # TODO: Implement pruning method
        # Unclear on how the CART algorithm post-prunes compared to C4.5
        raise NotImplementedError

    @staticmethod
    def _best_split_classification(feature_values, labels, data_type, classes, metric_func):
        """
         Determines the best split possible for a given feature.
         Helper method for grow_tree()
         :param feature_values, np.array, subset of variables for given feature to split on
         :param labels, np.array, subset of classes
         :param data_type, str, data type (i.e. numeric, ordinal, or categorical) for given feature
         :param classes, np.array, array of all possible class values
         :returns tuple, (best_thr, impurity, best_left_dist, best_right_dist)
                  Returns best value to split on and associated impurity,
                  along with the class distributions in each node.
         """

        best_thr, impurity, best_left_dist, best_right_dist = None, 1, None, None   # current min impurity

        #sort data
        indx = np.argsort(feature_values, axis=0).reshape(-1)
        feature_values, labels = feature_values[indx], labels[indx]

        possible_thresholds = np.unique(feature_values)

        num_labels = labels.size

        if data_type == 'c':
            possible_thresholds = find_splits(possible_thresholds)

        for threshold in possible_thresholds:

            if data_type == 'c':
                selection = np.isin(feature_values, threshold)
            else:
                selection = feature_values > threshold

            right = labels[selection]
            left = labels[~selection]

            num_right = right.size

            # compute distribution of labels for each split
            unique_right, right_distribution = np.unique(right, return_counts=True)
            unique_left, left_distribution = np.unique(left, return_counts=True)

            # assure class distributions are in the correct order and the correct shape
            new_right, new_left = np.zeros(classes.shape), np.zeros(classes.shape)
            inx_right = np.isin(classes, unique_right, assume_unique=True)
            inx_left = np.isin(classes, unique_left, assume_unique=True)
            new_right[inx_right], new_left[inx_left] = right_distribution, left_distribution

            right_distribution, left_distribution = new_right, new_left

            # compute impurity of split based on the distribution
            gini_right = metric_func(np.array(right_distribution), num_right)
            gini_left = metric_func(np.array(left_distribution), num_labels - num_right)

            # compute weighted total impurity of the split
            gini_split = (num_right * gini_right + (num_labels - num_right) * gini_left) / num_labels

            if gini_split < impurity:
                best_thr, impurity, best_left_dist, best_right_dist = threshold, gini_split, left_distribution, right_distribution

        # returns the threshold with the min associated impurity value --> best split threshold
        return best_thr, impurity, best_left_dist, best_right_dist

    def __repr__(self):
        return f"DecisionTreeClassification(name='{self.name}', children={[child for child in self.children]})"

    def __str__(self):
        return self.__repr__()


class KeDTClassification(GeneralDecisionTree):
    """
     Decision Tree implemented to support method that utilizes a symetric KL-Divergence to avoid
     iterating over data.
     The only important differences from the previous class is in _best_split_classification
    """

    def __init__(self, class_counts, n_subset, name='root',
                 children=None, split_rule=None, split_feature=None):
        """
        Initialize decision tree with functionality specific for traditional classification.
        :param class_counts, np.array, counts of examples for each class in node
        :param n_subset, int, number of examples represented by this node
        :param name, str, (default='root'), see parent class
        :param children, list, (default=None), see parent class
        :param split_rule, lambda function, (default=None), function that returns index of child for split
        :param split_feature, int, (default=None), index of feature to split on in dataset for current node
        """

        super().__init__(name=name, children=children,
                         split_rule=split_rule, split_feature=split_feature)

        self.class_counts = class_counts
        self.n_subset = n_subset

    def grow_tree(self, X, y, data_types, best_gini, classes, metric_func=gini,
                  min_size=2, max_depth=None, current_depth=0, max_gini=1):
        """
        Grows tree for given dataset for a classification task.  Recursive.
        :param X, np.array, subset of variables to split on
        :param y, np.array, subset of classes
        :param data_types, list, data types (i.e. numeric, ordinal, or categorical) of X
        :param best_gini, double, gini of the current node being split
        :param classes, np.array, array of all possible class values
        :param metric_func, function,  (default = "gini"), type of splitting used, either "gini" or "entropy"
        :param min_size, int, (default=2) minimum allowable number of examples making up a node
        :param max_depth, int, (default=None) maximum number of branches off nodes allowed
        :param current_depth, used in recursion to keep track of tree depth
        :param max_gini, int, (default=1) maximum gini you allow for a split to happen
        """
        assert all(data_t == 'n' for data_t in data_types), "KeDTClassification currently only works for numeric data"
        assert classes.size == 2, "KeDTClassification currently only works for binary classification"

        _, class_pop = np.unique(y, return_counts=True)

        if (y.size < min_size) or (best_gini == 0.0) or class_pop[0]<2 or class_pop[1]<2:
            # stopping criterion: node be smaller than min size
            # if node is pure, don't split
            return
        if max_depth is not None:
            if current_depth > max_depth:
                # stopping criterion: cannot build tree greater than max size
                return

        best_thr, best_p_ind, best_type = None, None, None

        for idx, data_type in zip(range(X.shape[1]), data_types):
            x = X[:, idx]

            new_thr, new_gini, left_distribution, right_distribution = self._best_split_classification(x, y, data_type,
                                                                                                       classes, metric_func=metric_func)

            if new_gini < best_gini:  # minimize gini
                best_gini, best_thr, best_p_ind, best_type, best_left_dist, best_right_dist = new_gini, new_thr, idx, data_type, left_distribution, right_distribution

        # if better split found
        if (best_thr is not None) and (max_gini > best_gini):

            # set current tree values for split
            self.split_feature = best_p_ind
            if best_type in ['n', 'o']:
                self.split_rule = lambda x_val: (x_val > best_thr).astype(int)
            else:
                self.split_rule = lambda x_val: (x_val == best_thr).astype(int)

            # calculate class distributions for children

            splits = self.split_rule(X[:, best_p_ind])

            # subset data for splits
            right_y, right_x = y[splits == 1],  X[splits == 1, :]
            left_y, left_x = y[splits == 0], X[splits == 0, :]

            if (right_y.size < min_size) or (left_y.size < min_size):
                # stopping criterion: if either child is less than min size, don't split
                self.split_rule = None
                self.split_feature = None
                return

            # grow left child

            left_tree = KeDTClassification(name=f"{self.name}_{best_p_ind}_child1",
                                                   class_counts=np.array(best_left_dist),
                                                   n_subset=np.sum(best_left_dist))

            left_tree.grow_tree(left_x, left_y, data_types, metric_func(np.array(best_left_dist), left_y.size),
                                classes=classes, min_size=min_size, max_depth=max_depth,
                                current_depth=current_depth + 1, metric_func=metric_func)

            # grow right child

            right_tree = KeDTClassification(name=f"{self.name}_{best_p_ind}_child2",
                                                    class_counts=np.array(best_right_dist),
                                                    n_subset=np.sum(best_right_dist))

            right_tree.grow_tree(right_x, right_y, data_types, metric_func(np.array(best_right_dist), right_y.size),
                                 classes=classes, min_size=min_size, max_depth=max_depth,
                                 current_depth=current_depth + 1, metric_func=metric_func)

            # add children to tree
            self.children = [left_tree, right_tree]

        else:
            # stopping criterion: gini not improved
            # gini is greater than user-specified maximum gini
            # gini is greater than user-specified maximum gini
            return

    def prune(self):
        # TODO: Implement pruning method
        # Unclear on how the CART algorithm post-prunes compared to C4.5
        raise NotImplementedError

    @staticmethod
    def _best_split_classification(feature_values, labels, data_type, classes, metric_func):
        """
        See base class for info
        """

        best_thr, impurity, best_left_dist, best_right_dist = None, 1, None, None   # current min impurity
        num_labels = labels.size

        min = np.min(feature_values)
        max = np.max(feature_values)
        step = (max - min) / 100  # currently always using 100 steps
        x_vals = np.arange(min, max, step)

        x_class_1 = feature_values[(labels == 1).flatten()]
        x_class_0 = feature_values[(labels == 0).flatten()]

        kernel_1 = gaussian_kde(x_class_1)
        kernel_0 = gaussian_kde(x_class_0)

        lambda_div = x_class_1.size / feature_values.size

        probs_1 = kernel_1(x_vals)
        probs_0 = kernel_0(x_vals)

        with np.errstate(divide='ignore', invalid='ignore'):       # a bandaid for a larger problem
            D_l1 = probs_1 * np.log2(probs_1 / (lambda_div * probs_1 + (1 - lambda_div) * probs_0))
            D_l0 = probs_0 * np.log2(probs_0 / (lambda_div * probs_1 + (1 - lambda_div) * probs_0))

        D_lall = lambda_div * D_l1 + (1 - lambda_div) * D_l0

        top_n = 5

        trimmed_D = D_lall[top_n:(D_lall.size - top_n)]
        trimmed_x_vals = x_vals[top_n:(x_vals.size - top_n)]
        ind_min = np.argpartition(trimmed_D, top_n-1)[0:top_n-1]

        top_5_partitions = trimmed_x_vals[ind_min]

        for threshold in top_5_partitions:

            selection = feature_values > threshold

            right = labels[selection]
            left = labels[~selection]

            num_right = right.size

            # compute distribution of labels for each split
            unique_right, right_distribution = np.unique(right, return_counts=True)
            unique_left, left_distribution = np.unique(left, return_counts=True)

            # assure class distributions are in the correct order and the correct shape
            new_right, new_left = np.zeros(classes.shape), np.zeros(classes.shape)
            inx_right = np.isin(classes, unique_right, assume_unique=True)
            inx_left = np.isin(classes, unique_left, assume_unique=True)
            new_right[inx_right], new_left[inx_left] = right_distribution, left_distribution

            right_distribution, left_distribution = new_right, new_left

            # compute impurity of split based on the distribution
            gini_right = metric_func(np.array(right_distribution), num_right)
            gini_left = metric_func(np.array(left_distribution), num_labels - num_right)

            # compute weighted total impurity of the split
            gini_split = (num_right * gini_right + (num_labels - num_right) * gini_left) / num_labels

            if gini_split < impurity:
                best_thr, impurity, best_left_dist, best_right_dist = threshold, gini_split, left_distribution, right_distribution

        # returns the threshold with the min associated impurity value --> best split threshold
        return best_thr, impurity, best_left_dist, best_right_dist

    def __repr__(self):
        return f"KeRFClassification(name='{self.name}', children={[child for child in self.children]})"

    def __str__(self):
        return self.__repr__()