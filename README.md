
<img src="doc/_static/logo.png" width="25%" height="25%" align="right" />

# Scikit-Fairness

The goal of this project is to attempt to consolidate fairness
related metrics, transformers and models into a package that (hopefully)
will become a contribution project to scikit-learn.

Fairness, in data science, is a complex unsolved problem for which many
tactics are proposed - each with their own advantage and disadvantages.
This packages aims to make these tactics readily available,
therefore enabling users to try and evaluate different fairness techniques.

Consider all the steps in a machine learning pipeline.

![](doc/_static/steps.png)

This package will offer tools at every step to make the pipeline more fair.

## Documentation

The documentation for this project can be found [here](https://scikit-fairness.netlify.app/).

## Data

We have datasets available that will help you benchmark your fairness tools.

- `skfair.datasets.load_arrests`

## Pre Processing

We have filtering techniques that try to filter out information that correlates
with sensitive attributes.

- `skfair.preprocessing.InformationFilter`

## Model

We have models you're able to constrain with regards to a fairness metric.

- `skfair.linear_model.DemographicParityClassifier`
- `skfair.linear_model.EqualOpportunityClassifier`

## Post Processing

We have meta estimators that allow you to correct the model after it has been trained.

## Measure

We offer metrics that are designed to measure unfairness in your dataset.

- `skfair.metrics.equal_opportunity_score`
- `skfair.metrics.p_percent_score`
