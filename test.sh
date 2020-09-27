set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length 100
mypy -m primer --strict
python -c 'import primer'
