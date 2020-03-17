import os
from setuptools import setup, find_packages

import skfair

base_packages = [
    "numpy>=1.16.0",
    "scipy>=1.2.0",
    "scikit-learn>=0.20.2",
    "pandas>=0.23.4",
    "patsy>=0.5.1",
    "autograd>=1.2",
    "cvxpy>=1.0.24",
    "Deprecated>=1.2.6",
]
docs_packages = [
    "sphinx>=1.8.5",
    "sphinx_rtd_theme>=0.4.3",
    "nbsphinx>=0.4.2",
    "ghp-import>=0.5.5"
]
test_packages = [
    "flake8>=3.6.0",
    "nbval>=0.9.1",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "pytest-mock>=1.6.3",
    "pre-commit>=1.18.3",
]
util_packages = [
    "matplotlib>=3.0.2",
    "plotnine>=0.5.1",
    "jupyter>=1.0.0",
    "jupyterlab>=0.35.4",
]
dev_packages = docs_packages + test_packages + util_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scikit-fairness",
    version=skfair.__version__,
    description="a collection of utilities to make algorithms more fair",
    author="Vincent D. Warmerdam & Matthijs Brouns",
    url="https://scikit-lego.readthedocs.io/en/latest/",
    packages=find_packages(exclude=["notebooks"]),
    package_data={"skfair": ["data/*.zip"]},
    install_requires=base_packages,
    extras_require={"docs": docs_packages, "dev": dev_packages, "test": test_packages},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
