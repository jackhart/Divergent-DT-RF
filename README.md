# Final Project Repo


## Setting Up Environment

Follow these steps to have the proper environment for these scripts.

1. Create a Python virtual environment called "anly601_finalproject" using `conda`

``` bash
$ conda create -n anly601_finalproject pip python=3.6
$ conda activate anly601_finalproject
```

2. Clone the repo and install required packages

``` bash
$ git clone https://github.com/jackhart/Kernel_Implementations_Random_Forests
$ pip install -r requirements.txt
```

3. Set up ModelsML package with setup.py 

``` bash
$ python setup.py develop
```



## Dataset

Working with data from [UCI Machine Learning Repo](https://archive.ics.uci.edu/ml/datasets.php)

## Command Line Examples

Example usage of `experiment_script.py`.

The Hparam object is used to abstract away *experiment parameters* and *model parameters*.  Their defaults are defined in `ModelsML/defined_params.py`. 

Here's an example usage of specific Hparam objects for the Donut dataset for a classic and Lambda-Divergent decision tree.


``` bash
$ python experiment_script.py --experiment_hparams classic_donut \
--model_hparams ClassicDecisionTreeClassifier_default # classic DT
```

``` bash
$ python experiment_script.py --experiment_hparams classic_donut \
--model_hparams KeDTClassifier_default # divergent DT
```

You can adjust specific details of an Hparams object in this script by indicating the changes in a comma-separated list.  This example adjusts the above code to instead use the breast cancer data.
 
 ``` bash
$ python experiment_script.py --experiment_hparams classic_donut \
--model_hparams KeDTClassifier_default \
--experiment_hparams_update dataset=breast_cancer
```