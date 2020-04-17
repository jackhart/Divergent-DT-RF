"""
4/16/20 -- Norman Hong --> Only decision tree implemented. Random Forests need to be debugged and tested.
Program implements decision tree algorithm for continuous or binary output variables.
Input variables are continuous or discrete.
"""
import math
import numpy as np
from sklearn.datasets import load_iris
from pprint import pprint
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
import random


# noinspection PyDefaultArgument
class DecisionTree(object):
    """
    CART Implementation:
     ftp://ftp.boulder.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/TREE-CART.pdf
    """

    # TODO: Keeping it simple by allowing for control of 1 model parameter.
    def __init__(self, max_depth=4, k_gamma=1, delta=.2, use_kernel=False, is_classification=False):
        self.train_sub = {}
        self.depth = 0
        self.max_depth = max_depth
        self.tree = {}
        self.gamma = k_gamma  # gamma parameter for kernel transform. This a weight term in radial basis kernel.
        self.delta = delta  # Parameter used to determine how close points should be to the boundary.
        self.leaf_num = 0  # Running count of the number of leafs created
        self.use_kernel = use_kernel  # Determine if should use kernel prediction or not.
        self.is_classification = is_classification  # Determine if classification tree or regression.

    @staticmethod
    def _entropy_function(c, n):
        """
        This is the math equation for entropy of ith class in a node.
        :param c: Number of points for a class in node.
        :param n: Total number of points in node.
        :return: class entropy
        """
        return -(c * 1.0 / n) * math.log(c * 1.0 / n, 2)

    @staticmethod
    def _entropy_cal(c1, c2):
        """
        Returns entropy of a node.
        :param c1: number of points for class 0
        :param c2: number of points for class 1
        :return: entropy of node
        """
        # When there is only one class in the group, entropy is 0.
        if c1 == 0 or c2 == 0:
            return 0
        else:
            return DecisionTree._entropy_function(c1, c1 + c2) + DecisionTree._entropy_function(c2, c1 + c2)

    @staticmethod
    def _entropy_of_one_division(division):
        """
        Returns entropy of a child node
        :param division: child node
        :return: entropy e and number of data points in node.
        """
        n = len(division)  # Number of points.
        classes = set(division)  # distinct classes.
        if len(division) == 0:  # If there is nothing in node. return 0 entropy
            return 0, 0
        else:
            one = classes.pop()
            e = DecisionTree._entropy_cal(sum(division == one), sum(division != one))
        return e, n

    def _get_criterion(self, y_predict, y_real):
        """
        Return entropy of a split.

        :param y_predict: this is the split decision. True or False. It determines if point belongs to right or left
        child node.
        :param y_real: These are the points.
        :return:
        """
        if len(y_predict) != len(y_real):
            print("y_pred and y_real have to be same length")
            return None
        n = len(y_real)

        if self.is_classification:  # if classification tree, then use entropy. Else use sum of squared errors.
            s_true, n_true = DecisionTree._entropy_of_one_division(y_real[y_predict])  # entropy of left child node.
            s_false, n_false = DecisionTree._entropy_of_one_division(y_real[~y_predict])  # entropy of right child node.
        else:
            s_true, n_true = DecisionTree._sum_of_squared_errors(y_real[y_predict])  # left child sse
            s_false, n_false = DecisionTree._sum_of_squared_errors(y_real[~y_predict])  # right child sse.
        s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false  # Overall entropy or sum of squared errors.
        return s

    def _find_best_split(self, col, y):
        """
        Iterates through all the unique values of a feature and identify the best split condition.
        :param col: single input variable.
        :param y: target variable.
        :return: minimum entropy or sse and cutoff point.
        """
        min_criterion = np.infty
        cut_off = None

        # Iterating through each data point in the column.
        for value in set(col):
            y_predict = col < value  # Separating the data points into two groups based on value variable.
            current = self._get_criterion(y_predict, y)
            if current <= min_criterion:
                min_criterion = current
                cut_off = value
        return min_criterion, cut_off

    def find_best_split_of_all(self, x, y):
        """
        Find the best split conditions for all features.
        :param x: all input variables. SHould be a matrix.
        :param y: target variable
        :return: the column to split on, cutoff value, and entropy.
        """
        index = None
        min_criterion = np.infty
        best_cutoff = None

        # Iterating through every feature. i is index of column and c is list of values of column i.
        for i, c in enumerate(x.T):
            # print(i)  # TODO: For debugging.
            value, curr_cutoff = self._find_best_split(c, y)
            # checking if the cutoff is perfect.
            if value == 0:
                return i, curr_cutoff, value
            # check if current split is better than current best split.
            elif value <= min_criterion:
                min_criterion = value
                best_cutoff = curr_cutoff
                index = i
        return index, best_cutoff, min_criterion

    @staticmethod
    def _sum_of_squared_errors(y):
        """
        Sum of squared errors equation
        :param y: np.array of actual values.
        :return: out: sum of squared errors
        """
        n = len(y)
        if n == 0:
            return 0, 0
        target = np.mean(y)
        diff = y - target  # Element-wise subtraction
        squared = diff ** 2  # Element-wise squared
        out = sum(squared)
        return out, n

    def _fit_classification(self, x, y, feature_names, par_node={}, depth=0):
        """
        x: Feature set, Column index should match index of feature_names
        y: target variable
        par_node: will be the tree generated for this x and y.
        depth: the depth of the current layer. Used to keep track of base case condition.
        """
        if par_node is None:  # base case 1: tree stops at previous level. par_node = {} is not None.
            return None
        elif len(y) == 0:  # base case 2: no data in this group.
            return None
        # If all observations belong to same class, then returns class.
        elif self._all_same(y):  # base case 3: all y is the same in this group. Leaf case.
            self.leaf_num += 1
            leaf = {
                'val': y[0],
                'index': self.leaf_num
            }
            self.train_sub[str(leaf)] = {
                'sub_x': [],
                'sub_y': []
            }
            return leaf
        elif depth >= self.max_depth:  # base case 4: max depth reached. don't split anymore, so make leaf node.
            self.leaf_num += 1
            leaf = {
                'val': np.round(np.mean(y)),
                'index': self.leaf_num
            }
            self.train_sub[str(leaf)] = {
                'sub_x': [],
                'sub_y': []
            }
            return leaf
        else:  # Recursively generate trees!
            # find one split given an information gain
            col, cutoff, entropy = self.find_best_split_of_all(x, y)
            y_left = y[x[:, col] < cutoff]  # left hand side data
            y_right = y[x[:, col] >= cutoff]  # right hand side data
            par_node = {
                'col': feature_names[col], 'index_col': col, 'cutoff': cutoff, 'val': np.round(np.mean(y)),
                'left': self._fit_classification(x[x[:, col] < cutoff], y_left, feature_names, {}, depth + 1),
                'right': self._fit_classification(x[x[:, col] >= cutoff], y_right, feature_names, {}, depth + 1)
            }  # finding the dominant class.
            # generate tree for the left hand side data
            # right hand side trees
            self.depth += 1  # increase the depth when node is divided. self.depth increases during
            # unrolling of recursion.
            self.tree = par_node
            return par_node

    def _fit_regression(self, x, y, feature_names, par_node={}, depth=0):
        """
        x: Feature set, Column index should match index of feature_names
        y: target variable
        par_node: will be the tree generated for this x and y.
        depth: the depth of the current layer. Used to keep track of base case condition.
        """
        if par_node is None:  # base case 1: tree stops at previous level. par_node = {} is not None.
            return None
        elif len(y) == 0:  # base case 2: no data in this group.
            return None
        # If all observations belong to same class, then returns class.
        elif self._all_same(y):  # base case 3: all y is the same. Leaf case.
            self.leaf_num += 1
            leaf = {
                'val': y[0],
                'index': self.leaf_num
            }
            self.train_sub[str(leaf)] = {
                'sub_x': [],
                'sub_y': []
            }
            return leaf
        elif depth >= self.max_depth:  # base case 4: max depth reached. don't split anymore, so make leaf node.
            self.leaf_num += 1
            leaf = {
                'val': np.mean(y + 1),
                'index': self.leaf_num
            }
            self.train_sub[str(leaf)] = {
                'sub_x': [],
                'sub_y': []
            }
            return leaf
        else:  # Recursively generate trees!
            # find one split given an information gain
            col, cutoff, sse = self.find_best_split_of_all(x, y)
            y_left = y[x[:, col] < cutoff]  # left hand side data
            y_right = y[x[:, col] >= cutoff]  # right hand side data
            par_node = {
                'col': feature_names[col], 'index_col': col, 'cutoff': cutoff, 'val': np.mean(y + 1),
                'left': self._fit_regression(x[x[:, col] < cutoff], y_left, feature_names, {}, depth + 1),
                'right': self._fit_regression(x[x[:, col] >= cutoff], y_right, feature_names, {}, depth + 1)
            }  # finding the dominant class.
            # generate tree for the left hand side data
            # right hand side trees
            self.depth += 1  # increase the depth when node is divided. self.depth increases during
            # unrolling of recursion.
            self.tree = par_node
            return par_node

    def fit(self, x, y, feature_names):
        if self.is_classification:
            par_node = self._fit_classification(x, y, feature_names, par_node={}, depth=0)
        else:
            par_node = self._fit_regression(x, y, feature_names, par_node={}, depth=0)
        return par_node

    @staticmethod
    def _all_same(items):
        return all(x == items[0] for x in items)

    def predict(self, x):
        results = np.array([0.0] * len(x))
        # For each row in test data.
        for i, c in enumerate(x):
            if self.use_kernel:
                out = self._kernel_predict(c)
                results[i] = out
            else:
                leaf_node = self._get_prediction(c)
                results[i] = leaf_node.get("val")
        return results

    def _get_prediction(self, row):
        """
        Returns leave that row belongs too.
        :param row: record to be predicted.
        :return: cur_layer: a leaf node.
        """
        cur_layer = self.tree  # get the tree we build in training
        # if not leaf node
        while cur_layer.get('cutoff'):
            # determining which child to go to.
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']

        # If leaf node, return node. In other words, return the node the record belongs to.
        return cur_layer

    def _training_subset_evaluator(self, row):
        """
        Determine if training instance is near the corners of hyperrectangular edges. The reason is because we're only
        recording training instances when all their components are near the decision boundary.

        :param: row: Single instaance from a training set.
        :return: row: if near decision boundary. Otherwise, return None.
        """
        cur_layer = self.tree  # get the tree we build from calling fit.
        # if not leaf node
        while cur_layer.get('cutoff'):
            # determining which child to go to.
            # diff is the percent distance from cutoff.
            diff = abs(row[cur_layer['index_col']] - cur_layer['cutoff']) / cur_layer['cutoff']
            if row[cur_layer['index_col']] < cur_layer['cutoff'] and diff < self.delta:
                cur_layer = cur_layer['left']
            elif row[cur_layer['index_col']] >= cur_layer['cutoff'] and diff < self.delta:
                cur_layer = cur_layer['right']
            else:  # Return None if not corner point.
                return None
        # If leaf node and point is near decision boundary return point.
        return row

    def _training_subset_evaluator_gen(self, row):
        """
        Determine if training instance is near the edge of a boundary.
        :param row: Single instance from a training set.
        :return: row: if near an edge, return record. Otherwise, return None.
        """
        cur_layer = self.tree
        counter = False
        while cur_layer.get('cutoff'):
            # determining which child to go to.
            # Loop ends when its a leaf node.
            # diff is the percent distance from cutoff.
            diff = abs(row[cur_layer['index_col']] - cur_layer['cutoff']) / cur_layer['cutoff']
            if diff < self.delta:  # If point is boundary point.
                counter = True
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            elif row[cur_layer['index_col']] >= cur_layer['cutoff']:
                cur_layer = cur_layer['right']
        # If leaf node and point is near decision boundary return point, then return leaf node. Else, return None.
        if counter:
            return cur_layer
        else:
            return None

    def load_training_subset(self, x, y):
        """
        Run this method to save subset of training data. This is then used in kernel_prediction method.
        :param x: Training feature space
        :param y: Training targets
        """
        for i, c in enumerate(x):  # Iterating through every row
            node = self._training_subset_evaluator_gen(c)
            if node is None:  # node is None if c is not a decision boundary point. If it isn't, move to next iteration.
                continue
            self.train_sub[str(node)]['sub_x'].append(c)
            self.train_sub[str(node)]['sub_y'].append(y[i])

    def _radial_kernel(self, x1, x2):
        """
        Vectorized radial basis kernel implementation.
        :param x1: data point 1
        :param x2: data point 2
        :return: K(x1, x2) = radial basis function transformation
        """
        diff = x1 - x2
        dot_prod = np.dot(diff, diff)
        k = math.exp(-self.gamma * dot_prod)
        return k

    def _kernel_predict(self, t_row):
        """
        Prediction on new point using kernelized weighting. New records travel down the decision tree until land into
        leaf node. Then training instances in leaf node are used in kernel weighted output.
        :param: t_row: Input data to be predicted.
        :return: out: Target Class.
        """
        if self.train_sub is None:
            raise Exception("Need to load training data.")
        node = self._get_prediction(t_row)
        train_x = np.array(self.train_sub[str(node)]['sub_x'])
        train_y = np.array(self.train_sub[str(node)]['sub_y'])
        num = 0
        denom = 0
        for i, c in enumerate(train_x):  # Iterating through every row. Index i corresponds to ith index in train_y.
            k_sim = self._radial_kernel(t_row, c)
            num += k_sim * train_y[i]
            denom += k_sim
        if denom == 0:  # If denominator is 0, which is also the case when there are no training instances.
            return node['val']  # Don't use kernel weighting prediction.
        out = num / denom

        if self.is_classification:
            return np.round(out)
        else:
            return out


class RandomForests(DecisionTree):
    def __init__(self, num_pred, num_tree, max_depth=100, k_gamma=1, delta=.2, use_kernel=False,
                 is_classification=False):
        """
        In random forest, trees grow until leafs are pure, bootstrap samples are used in each iteration, and random
        subset of predictor variables are created in each node split.
        :param num_pred:
        :param num_tree:
        :param max_depth:
        :param k_gamma:
        :param delta:
        :param use_kernel:
        :param is_classification:
        """
        super().__init__(max_depth=max_depth,
                         k_gamma=k_gamma,
                         delta=delta,
                         use_kernel=use_kernel,
                         is_classification=is_classification)
        self.num_pred = num_pred  # Number of predictors to use for each split.
        self.num_tree = num_tree  # Number of trees to create in the ensemble.
        self.trees = []  # Collection of trees.

    def find_best_split_of_all(self, x, y):
        """
        Find the best split conditions for all features.
        :param x: all input variables. SHould be a matrix.
        :param y: target variable
        :return: the column to split on, cutoff value, and entropy.
        """
        index = None
        min_criterion = np.infty
        best_cutoff = None
        sub = random.sample(range(1, len(x.T)), self.num_pred)  # Generate index for subset of predictors of x.

        # Iterating through every feature. i is index of column and c is list of values of column i.
        for i, c in enumerate(x.T[sub]):
            # print(i)  # TODO: For debugging.
            value, curr_cutoff = self._find_best_split(c, y)
            # checking if the cutoff is perfect.
            if value == 0:
                return i, curr_cutoff, value
            # check if current split is better than current best split.
            elif value <= min_criterion:
                min_criterion = value
                best_cutoff = curr_cutoff
                index = i
        return index, best_cutoff, min_criterion

    @staticmethod
    def _get_bootstrap_sample(x, y):
        n_obs = len(x)
        index = random.choices(range(1, n_obs), n_oba)  # Create random sample with same number of obs as x with replacement.
        return x[index], y[index]

    def fit(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        for i in range(self.num_tree):
            bs_x, bs_y = self._get_bootstrap_sample(x, y)
            if self.is_classification:
                dt = self._fit_classification(bs_x, bs_y, feature_names, par_node={}, depth=0)
            else:
                dt = self._fit_regression(bs_x, bs_y, feature_names, par_node={}, depth=0)
            self.trees.append(dt)
        return self.trees

    # TODO: predict function and load data function.
    def predict(self, x):
        results = np.array([0.0] * len(x))
        # For each row in test data.
        for i, c in enumerate(x):
            temp = np.array[0.0] * len(self.num_tree)
            for j, tree in enumerate(self.trees):  # Iterating through all trees in forest.
                self.tree = tree  # Hot-fix to make self._get_prediction() to work.
                if self.use_kernel:
                    out = self._kernel_predict(c)
                    temp[j] = out
                else:
                    leaf_node = self._get_prediction(c)
                    temp[j] = leaf_node.get("val")
            results[i] = np.mean(temp)
        return results

    def load_training_subset(self, x, y):
        """
        Run this method to save subset of training data. This is then used in kernel_prediction method.
        :param x: Training feature space
        :param y: Training targets
        """
        for i, c in enumerate(x):  # Iterating through every row
            for j, tree in enumerate(self.trees):  # Iterate through all trees in forest.
                self.tree = tree  # Hot-fix so self._training_subset_evaluato_gen() works.
                node = self._training_subset_evaluator_gen(c)
                if node is None:  # node is None if c is not a decision boundary point. If it isn't, move to next iteration.
                    continue
                self.train_sub[str(j) + " " + str(node)]['sub_x'].append(c)
                self.train_sub[str(j) + " " + str(node)]['sub_y'].append(y[i])

    def _kernel_predict(self, t_row):
        """
        Prediction on new point using kernelized weighting. New records travel down the decision tree until land into
        leaf node. Then training instances in leaf node are used in kernel weighted output.
        :param: t_row: Input data to be predicted.
        :return: out: Target Class.
        """
        if self.train_sub is None:
            raise Exception("Need to load training data.")
        node = self._get_prediction(t_row)
        temp = np.array([0.0] * len(self.num_tree))
        for j in range(self.num_tree):
            train_x = np.array(self.train_sub[str(j) + " " + str(node)]['sub_x'])
            train_y = np.array(self.train_sub[str(j) + " " + str(node)]['sub_y'])
            num = 0
            denom = 0
            for i, c in enumerate(train_x):  # Iterating through every row. Index i corresponds to ith index in train_y.
                k_sim = self._radial_kernel(t_row, c)
                num += k_sim * train_y[i]
                denom += k_sim
            if denom == 0:  # If denominator is 0, which is also the case when there are no training instances.
                temp[j] = node['val']  # Don't use kernel weighting prediction.
            out = num / denom
            temp[j] = out
        if self.is_classification:
            return np.round(np.mean(temp))
        else:
            return np.mean(temp)


def get_accuracy(y_predict, y_real):
    n_obs = len(y_predict)
    error = sum(abs(y_predict - y_real)) / n_obs
    # pprint("Accuracy: {}".format(1 - error))
    return 1 - error


def get_mse(y_predict, y_real):
    n_obs = len(y_predict)
    error = (y_predict - y_real) ** 2  # Element-wise square.
    mse = sum(error) / n_obs
    # pprint("Mean squared error: {}".format(mse))
    return mse


def cross_validation(model, x, y, feature_names):
    random.seed(100)
    folds = 10  # Number of validation sets to perform
    frac = 1 - (1 / folds)  # Fraction of train dataset
    num_records = len(x)
    train_size = np.round(num_records * frac)
    metric_list = []
    for iteration in range(folds):
        index = random.sample(range(num_records), k=int(train_size))
        train = x[index]
        validation = np.delete(x, index, axis=0)
        train_target = y[index]
        valid_target = np.delete(y, index, axis=0)

        model.fit(train, train_target, feature_names=feature_names)
        model.load_training_subset(train, train_target)
        y_pred = model.predict(validation)
        if model.is_classification:
            metric_list.append(get_accuracy(y_pred, valid_target))
        else:
            metric_list.append(get_mse(y_pred, valid_target))
    if model.is_classification:
        pprint("10 fold test accuracy error: {}".format(np.mean(metric_list)))
    else:
        pprint("10 fold test mse: {}".format(np.mean(metric_list)))


# For now, just run each line in main() to execute program.
def iris():
    iris = load_iris()
    x = iris.data
    y = iris.target
    clf = DecisionTree(max_depth=2, k_gamma=.5, delta=1, use_kernel=True, is_classification=True)
    cross_validation(clf, x, y, iris.feature_names)

    clf = DecisionTree(max_depth=2, k_gamma=.5, use_kernel=False, is_classification=True)
    cross_validation(clf, x, y, iris.feature_names)


def boston():
    iris = load_boston()
    x = iris.data
    y = iris.target

    clf = DecisionTree(max_depth=4, k_gamma=.00001, delta=.2, use_kernel=True, is_classification=False)
    cross_validation(clf, x, y, iris.feature_names)

    clf = DecisionTree(max_depth=4, k_gamma=.00001, delta=.2, use_kernel=False, is_classification=False)
    cross_validation(clf, x, y, iris.feature_names)


def diabetes():
    iris = load_diabetes()
    x = iris.data
    y = iris.target

    clf = DecisionTree(max_depth=6, k_gamma=.001, delta=.4, use_kernel=True, is_classification=False)
    cross_validation(clf, x, y, iris.feature_names)

    clf = DecisionTree(max_depth=6, k_gamma=.001, delta=.4, use_kernel=False, is_classification=False)
    cross_validation(clf, x, y, iris.feature_names)


def bc():
    iris = load_breast_cancer()
    x = iris.data
    y = iris.target

    clf = DecisionTree(max_depth=4, k_gamma=.00001, delta=.2, use_kernel=True, is_classification=True)
    cross_validation(clf, x, y, iris.feature_names)

    clf = DecisionTree(max_depth=4, k_gamma=.00001, delta=.2, use_kernel=False, is_classification=True)
    cross_validation(clf, x, y, iris.feature_names)


if __name__ == '__main__':
    iris()
    diabetes()
    boston()
    bc()
