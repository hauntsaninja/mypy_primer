set -ex

isort --diff --check --quiet .
black --diff --check --quiet .
flake8 --max-line-length=100 --ignore=E203,W503 $(git ls-files | grep "py$")
mypy -p mypy_primer --strict --python-version 3.10
# check we have unique projects
python -c 'from mypy_primer.projects import get_projects; get_projects()'
# this check was meant to ensure we could programmatically update the projects
# i've disabled for now, i think it's nice to be able to e.g. add inline comments
# we can re-enable if we remove that limitation or if we have a strong need
# for programmatic updates
# python -c 'from mypy_primer.projects import get_projects, update_projects; update_projects(get_projects(), check=True)'
