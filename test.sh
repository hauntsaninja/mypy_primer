set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length=100 --ignore=E203,W503 $(git ls-files | grep "py$")
mypy -m mypy_primer --strict
python -c 'import mypy_primer'
