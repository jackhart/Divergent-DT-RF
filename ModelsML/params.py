""" params class structure for custom experiments
    Basic design principles came from here: https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/
"""
from shlex import shlex


class Hparams(dict):
    """
    Simple wrapper of dictionary that allows access of values through attributes.
    Also implements update function for values given a command line string
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self.__dict__.update(d)

    def update_attributes(self, comma_str):
        """
        Update function for values within object, given a command line string
        :param comma_str: string of comma separated keys and their update values (e.g. 'seed=58,data_type=xor')
        """

        # create shlex object for parsing
        shlex_str = shlex(comma_str, posix=True)
        shlex_str.whitespace_split = True
        shlex_str.whitespace = ','

        # parse update values in dict
        update_values = dict(layers.split('=', 1) for layers in shlex_str)

        # update attributes in object
        for key, value in update_values.items():
            if self[key] is None:  # If None is default, assume float TODO: think of a better way to handle this
                current_type = float
            else:
                current_type = type(self[key])

            self[key] = current_type(value)  # cast to type of previous param


class ParamsContainer(dict):
    """
    This is a container for experiment hyper-parameters.
    Allows hyper-parameters to be saved using a decorator.

    Example Usage:

    @all_hparams.register('default_experiment')
    def define_config():
      config = Hparams()
      config.seed = 58
      ...
      return config
    """

    def register(self, key):
        """
        Update function for values given a command line string
        :param key: (optional), the name/key of the given Hparams object
        :returns function, the updated function which now adds an Hparams to this container
        """
        # decorator function that will save Hparams
        def decorator(value, key_name):
            hparams = value()  # run function
            assert isinstance(hparams, Hparams), "ParamsContainer can only hold Hparams objects"

            self[key_name] = hparams
            return

        return lambda value: decorator(value, key_name=key)


class ParamsContainers:
    """
    This is a static class used to save all ParamContainer objects.
    Example Usage:

    @ParamsContainers.experiment_params.register('test_params')
    def define_config():
      config = Hparams()
      config.seed = 58
      ...
      return config
    """
    def __init__(self):
        raise RuntimeError("ParamsContainers is a static class")

    model_params = ParamsContainer()
    experiment_params = ParamsContainer()
