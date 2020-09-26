set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length 100
mypy -m primer --strict --no-warn-return-any
python -c 'import primer'
