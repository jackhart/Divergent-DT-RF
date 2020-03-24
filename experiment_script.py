#!/usr/bin/env python

from ModelsML.DecisionTreeEstimators import ClassicDecisionTreeClassifier
from ModelsML.util import create_synthetic_data

import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main(data_file):
    """run experiments"""

    # create dataset
    xor_X, xor_y = create_synthetic_data(seed=55, n=100, type='xor')

    X_train, X_test, y_train, y_test = train_test_split(xor_X, xor_y, test_size=0.1, random_state=55)

    # create estimator
    basic_tree = ClassicDecisionTreeClassifier()
    basic_tree_fitted = basic_tree.train(X_train, y_train, data_types=['n', 'n'])

    # holdout example
    probabilities, predictions = basic_tree_fitted.predict(X_test, y_test)

    # print results
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment runner')
    parser.add_argument('--uci_data', type=str, default="ICI/data/breast-cancer-wisconsin.data",
                        help='path to dataset')

    args = parser.parse_args()
    main(args.uci_data)
