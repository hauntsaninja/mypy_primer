set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length=100 --ignore=E203,W503 $(git ls-files | grep "py$")
mypy -p mypy_primer --strict
# check we have unique projects
python -c 'from mypy_primer.projects import get_projects; get_projects()'
