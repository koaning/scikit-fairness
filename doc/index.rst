.. scikit-fairness documentation master file, created by
   sphinx-quickstart on Tue Mar 19 20:15:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-fairness
===============

.. image:: _static/logo.png

The goal of this project is to attempt to consolidate fairness
related metrics, transformers and models into a package that (hopefully)
will become a contribution project to scikit-learn.

Consider all the steps in a machine learning pipeline.

.. image:: _static/steps.png

This package will offer tools at every step.

Fairness, in data science, is a complex unsolved problem for which many
tactics are proposed - each with their own advantage and disadvantages.
This packages aims to make these tactics readily available,
therefore enabling users to try and evaluate different fairness techniques.


Disclaimer
**********

This package is not (yet, it is a goal) formally affiliated with scikit-learn.

Installation
************

Install `scikit-fairness` via pip with

.. code-block:: bash

   pip install scikit-fairness


Alternatively you can fork/clone and run:

.. code-block:: bash

   pip install --editable .


Usage
*****

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import Pipeline

   from skfairness.preprocessing import InformationFilter

   ...

   mod = Pipeline([
       ("information_filter", InformationFilter()),
       ("model", LogisticRegression(solver='lbfgs'))
   ])

   ...



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   contribution
   fairness_boston_housing.ipynb

   api/modules
