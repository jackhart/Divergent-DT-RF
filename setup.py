from setuptools import setup
from Cython.Build import cythonize

setup(name='ModelsML',
      version='0.1',
      ext_modules=cythonize("DecisionTrees.pyx"),
      description='Final Project For Advanced ML',
      author='Jack Hart',
      zip_safe=False)
