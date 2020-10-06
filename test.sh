set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length 100
mypy -m mypy_primer --strict
python -c 'import mypy_primer'
