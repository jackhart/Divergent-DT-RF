#!/usr/bin/env python

from ModelsML.DecisionTreeEstimators import ClassicDecisionTreeClassifier
from ModelsML.RandomForestEstimators import ClassicRandomForestClassifier
from ModelsML.util import create_synthetic_data_function, load_UCI_function, time_function
from ModelsML.defined_params import *

import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


classifiers = {'ClassicDecisionTreeClassifier': ClassicDecisionTreeClassifier,
               'ClassicRandomForestClassifier': ClassicRandomForestClassifier}

datasets = {'xor': create_synthetic_data_function(type_p='xor'),
            'donut': create_synthetic_data_function(type_p='donut'),
            'iris': create_synthetic_data_function(type_p='iris'),
            'wine': create_synthetic_data_function(type_p='wine'),
            'votes': load_UCI_function(type_p='votes')}

split_types = ['holdout']  # TODO: Add implementation beyond holdout


def main(args):
    """run experiments"""

    # extract hparams for experiment and model
    experiment_hparams = ParamsContainers.experiment_params[args.experiment_hparams]
    model_hparams = ParamsContainers.model_params[args.model_hparams]

    # update hparams
    if args.experiment_hparams_update is not None:
        experiment_hparams.update_attributes(args.experiment_hparams_update)
    if args.model_hparams_update is not None:
        model_hparams.update_attributes(args.model_hparams_update)

    # create dataset
    dataset_x, dataset_y, data_types = datasets[experiment_hparams.dataset](experiment_hparams)

    # instantiate estimator
    basic_tree = classifiers[model_hparams.model]()

    # Only Holdout implemented
    # TODO: Add functionality for cross validation in Hparams
    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y,
                                                        test_size=experiment_hparams.prop_test,
                                                        random_state=experiment_hparams.seed)

    # train estimator
    basic_tree_fitted, train_time = basic_tree.train(x_train, y_train, model_hparams, data_types=data_types)

    # predict
    probabilities_train, predictions_train, train_pred_time = basic_tree_fitted.predict(x_train)
    probabilities_test, predictions_test, test_pred_time = basic_tree_fitted.predict(x_test)

    # print results
    print(f"Train Accuracy: {accuracy_score(y_train, predictions_train)}")
    print(f"Test Accuracy: {accuracy_score(y_test, predictions_test)}")
    print(f"Train Time: {train_time}")
    print(f"Train Prediction Time: {train_pred_time}")
    print(f"Test Prediction Time: {test_pred_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment runner')

    """See defined_params.py for Hparam objects"""
    parser.add_argument('--experiment_hparams', type=str, default='classic_xor',
                        help='ParamContainer for experiment hyper-parameters')
    parser.add_argument('--model_hparams', type=str, default='ClassicDecisionTreeClassifier_default',
                        help='ParamContainer for experiment hyper-parameters')

    parser.add_argument('--experiment_hparams_update', type=str, default=None,
                        help='Comma separated key-value pairs to update Hparams object')
    parser.add_argument('--model_hparams_update', type=str, default=None,
                        help='Comma separated key-value pairs to update Hparams object')

    args = parser.parse_args()
    main(args)

