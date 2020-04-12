from .params import Hparams, ParamsContainers

"""Experiment hyper-parameters"""


@ParamsContainers.experiment_params.register('classic_xor')
def define_config():
  config = Hparams()
  config.seed = 58
  config.split_type = 'holdout'
  config.dataset = 'xor'
  config.prop_test = 0.1
  config.n = 500
  return config


@ParamsContainers.experiment_params.register('classic_donut')
def define_config():
  config = Hparams()
  config.seed = 58
  config.dataset = 'donut'
  config.split_type = 'holdout'
  config.prop_test = 0.1
  config.n = 500

  return config


@ParamsContainers.experiment_params.register('classic_votes')
def define_config():
  config = Hparams()
  config.seed = 58
  config.dataset = 'votes'
  config.split_type = 'holdout'
  config.data_path = "UCI/data/house-votes-84.data"
  config.prop_test = 0.1

  return config


"""Model hyper-parameters"""


@ParamsContainers.model_params.register('ClassicDecisionTreeClassifier_default')
def define_config():
  config = Hparams()
  config.model = 'ClassicDecisionTreeClassifier'
  config.min_size = 2
  config.max_depth = None
  config.max_gini = 1.0
  config.metric = "entropy"

  return config


@ParamsContainers.model_params.register('ClassicRandomForestClassifier_default')
def define_config():
  config = Hparams()
  config.model = 'ClassicRandomForestClassifier'
  config.min_size = 2
  config.max_depth = None
  config.max_gini = 1.0
  config.metric = "gini"

  config.seed = 58
  config.m_try = 1
  config.n_trees = 100
  config.bootstrap = 100

  return config

@ParamsContainers.model_params.register('KeDTClassifier_default')
def define_config():
  config = Hparams()
  config.model = 'KeDTClassifier'
  config.min_size = 2
  config.max_depth = None
  config.max_gini = 1.0
  config.metric = "entropy"

  return config

