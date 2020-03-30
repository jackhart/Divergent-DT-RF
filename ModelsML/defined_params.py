from .params import Hparams, ParamsContainers


@ParamsContainers.experiment_params.register('classic_xor')
def define_config():
  config = Hparams()
  config.seed = 58
  config.model = 'ClassicDecisionTreeClassifier'
  config.split_type = 'holdout'
  config.dataset = 'xor'
  config.prop_test = 0.1
  return config


@ParamsContainers.experiment_params.register('classic_donut')
def define_config():
  config = Hparams()
  config.seed = 58
  config.model = 'ClassicDecisionTreeClassifier'
  config.dataset = 'donut'
  config.split_type = 'holdout'
  config.prop_test = 0.1

  return config


@ParamsContainers.experiment_params.register('classic_iris')
def define_config():
  config = Hparams()
  config.seed = 58
  config.model = 'ClassicDecisionTreeClassifier'
  config.dataset = 'iris'
  config.split_type = 'holdout'
  config.prop_test = 0.1

  return config





