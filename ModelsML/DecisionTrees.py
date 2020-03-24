"""Implementations of Decision Trees"""

from .util import find_splits, gini
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

        if self.children is None:
            return self
        try:
            child_index = self.split_rule(x[self.split_feature])[0]
        except TypeError:
            return self  # edge case if test data contains NA

        return self.children[child_index].traverse(x)

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

    def __init__(self, class_counts, n_subset, name='root',
                 children=None, split_rule=None, split_feature=None):
        super().__init__(name=name, children=children,
                         split_rule=split_rule, split_feature=split_feature)

        self.class_counts = class_counts
        self.n_subset = n_subset

    def grow_tree(self, X, y, data_types, best_gini, classes,
                  min_size=2, max_depth=None, current_depth=0, max_gini=1):

        if (y.size < min_size) or (best_gini == 0.0):
            # stopping criterion: node be smaller than min size
            # if node is pure, don't split
            return
        if max_depth is not None:
            if current_depth > max_depth:
                # node be smaller than min size
                return

        best_thr, best_p_ind, best_type = None, None, None

        for idx, data_type in zip(range(X.shape[1]), data_types):
            x = X[:, idx]

            new_thr, new_gini = self._best_split_classification(x, y, data_type, classes)

            if new_gini < best_gini:  # minimize gini
                best_gini, best_thr, best_p_ind, best_type = new_gini, new_thr, idx, data_type

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

            right_y = y[splits == 1]
            left_y = y[splits == 0]

            if (right_y.size < min_size) or (left_y.size < min_size):
                # stopping criterion: if either child is less than min size, don't split
                self.split_rule = None
                return

            _, right_distribution = np.unique(right_y, return_counts=True)
            _, left_distribution = np.unique(left_y, return_counts=True)

            # grow left child
            left_tree = DecisionTreeClassification(name=f"{self.name}_{best_p_ind}_child1",
                                                   class_counts=np.array(left_distribution), n_subset=left_y.size)

            left_tree.grow_tree(X[splits == 0, :], left_y, data_types, gini(np.array(left_distribution), left_y.size),
                                classes=classes, min_size=min_size, max_depth=max_depth, current_depth=current_depth + 1)

            # grow right child
            right_tree = DecisionTreeClassification(name=f"{self.name}_{best_p_ind}_child2",
                                                    class_counts=np.array(right_distribution), n_subset=right_y.size)

            right_tree.grow_tree(X[splits == 1, :], right_y, data_types, gini(np.array(right_distribution), right_y.size),
                                 classes=classes, min_size=min_size, max_depth=max_depth, current_depth=current_depth + 1)

            # add children to tree
            self.children = [left_tree, right_tree]

        else:
            # stopping criterion: gini not improved
            # gini is greater than user-specified maximum gini
            return

    @staticmethod
    def _best_split_classification(feature_values, labels, data_type, classes):
        impurity = []
        possible_thresholds = np.unique(feature_values)

        num_labels = labels.size

        if data_type == 'c':
            possible_thresholds = find_splits(possible_thresholds)

        for threshold in possible_thresholds:

            if data_type == 'c':
                selection = np.isin(feature_values, threshold)
            else:
                selection = feature_values >= threshold

            right = labels[selection]
            left = labels[~selection]

            num_right = right.size

            # compute distribution of labels for each split
            unique_right, right_distribution = np.unique(right, return_counts=True)
            unique_left, left_distribution = np.unique(left, return_counts=True)

            # assure class distributions are in the correct order and the correct shape
            new_right, new_left = np.zeros(classes.shape), np.zeros(classes.shape)
            inx_right, inx_left = np.where(unique_right == classes), np.where(unique_left == classes)
            new_right[inx_right], new_left[inx_left] = right_distribution, left_distribution

            right_distribution, left_distribution = new_right, new_left

            # compute impurity of split based on the distribution
            gini_right = gini(np.array(right_distribution), num_right)
            gini_left = gini(np.array(left_distribution), num_labels - num_right)

            # compute weighted total impurity of the split
            gini_split = (num_right * gini_right + (num_labels - num_right) * gini_left) / num_labels

            impurity.append(gini_split)

            # Debug
            # print(f"right dist: {right_distribution}")
            # print(f"left dist: {left_distribution}")
            # print(f"right gini: {gini_right}")
            # print(f"left gini: {gini_left}")
            # print(f"gini split: {gini_split}")

        # Debug - Minimum gini
        # print(f"min gini: {np.min(impurity)}")

        # returns the threshold with the min associated impurity value --> best split threshold
        return possible_thresholds[np.argmin(impurity)], np.amin(impurity)

    def __repr__(self):
        return f"DecisionTreeClassification(name='{self.name}', children={[child for child in self.children]})"

    def __str__(self):
        return self.__repr__()
