set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length=100 --ignore=E203,W503 $(git ls-files | grep "py$")
mypy -p mypy_primer --strict --python-version 3.9
# check we have unique projects and the list is formatted correctly
python -c 'from mypy_primer.projects import get_projects, update_projects; update_projects(get_projects(), check=True)'
