.PHONY: docs

flake:
	flake8 skfair
	flake8 tests
	flake8 setup.py

install:
	pip install -e .

develop:
	pip install -e ".[dev]"
	pre-commit install
	python setup.py develop

doctest:
	python -m doctest -v skfair/*.py

test: doctest
	pytest --disable-warnings --cov=skfair
	rm -rf .coverage*
	pytest --nbval-lax doc/*.ipynb

precommit:
	pre-commit run

spelling:
	codespell skfair/*.py

notebook:
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=0 doc/fairness.ipynb --output doc/fairness.ipynb

docs:
	rm -rf doc/.ipynb_checkpoints
	sphinx-build doc docs

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf docs
	rm -rf scikit_fairness.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*
	rm -rf tests/**/__pycache__

black:
	black skfair tests setup.py

check: flake precommit test spelling clean

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

pages: docs
	netlify deploy --dir docs --prod --site 81dfb2c9-cf7f-44ec-b78f-4525e0bd11bf --open
