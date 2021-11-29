# mypy_primer

mypy_primer makes it easy to run [mypy](https://github.com/python/mypy/) over a few million lines of
open source projects for the purpose of finding regressions or evaluating changes.

## Explanation

Here's what mypy_primer does:
- Clones a copy of mypy (potentially from a fork you specified)
- Checks out a "new" and "old" revision of mypy
- Clones a hardcoded list of projects (potentially filtered by you)
- Installs necessary stubs and dependencies per project
- Runs the appropriate mypy command per project
- Shows you the potentially differing results!
- Lets you bisect to find the commit that causes a given change

mypy_primer contains a hardcoded list of open source projects and their respective mypy setups (to
which contributions are gladly accepted). The list is visible at the bottom of `mypy_primer.py` and
many of them should be recognisable names. I used https://grep.app to help me; a mypy.ini or "mypy"
in a tox.ini / setup.cfg / .travis.yml / etc is a pretty strong signal. This hardcoded list is in
theory susceptible to bitrot, but if you pass e.g. the flag `--project-date 2020-09-25` to
mypy_primer, it'll check out projects as they were today and things should work!

## Usage

```
λ mypy_primer --help
usage: mypy_primer [-h] [--new NEW] [--old OLD] [--repo REPO]
                   [--mypyc-compile-level MYPYC_COMPILE_LEVEL]
                   [--custom-typeshed-repo CUSTOM_TYPESHED_REPO] [--new-typeshed NEW_TYPESHED]
                   [--old-typeshed OLD_TYPESHED] [-k PROJECT_SELECTOR] [-p LOCAL_PROJECT]
                   [--expected-success] [--project-date PROJECT_DATE] [--num-shards NUM_SHARDS]
                   [--shard-index SHARD_INDEX] [-o {full,diff,concise}] [--old-success]
                   [--coverage] [--bisect] [--bisect-output BISECT_OUTPUT] [-j CONCURRENCY]
                   [--debug] [--base-dir BASE_DIR] [--clear]

optional arguments:
  -h, --help            show this help message and exit

mypy:
  --new NEW             new mypy version, defaults to HEAD (pypi version, anything commit-ish, or
                        isoformatted date)
  --old OLD             old mypy version, defaults to latest tag (pypi version, anything commit-
                        ish, or isoformatted date)
  --repo REPO           mypy repo to use (passed to git clone. if unspecified, we first try pypi,
                        then fall back to github)
  --mypyc-compile-level MYPYC_COMPILE_LEVEL
                        Compile mypy with the given mypyc optimisation level
  --custom-typeshed-repo CUSTOM_TYPESHED_REPO
                        typeshed repo to use (passed to git clone)
  --new-typeshed NEW_TYPESHED
                        new typeshed version, defaults to mypy's (anything commit-ish, or
                        isoformatted date)
  --old-typeshed OLD_TYPESHED
                        old typeshed version, defaults to mypy's (anything commit-ish, or
                        isoformatted date)

project selection:
  -k PROJECT_SELECTOR, --project-selector PROJECT_SELECTOR
                        regex to filter projects (matches against location)
  -p LOCAL_PROJECT, --local-project LOCAL_PROJECT
                        run only on the given file or directory. if a single file, supports a '#
                        flags: ...' comment, like mypy unit tests
  --expected-success    filter to hardcoded subset of projects where some recent mypy version
                        succeeded aka are committed to the mypy way of life. also look at: --old-
                        success
  --project-date PROJECT_DATE
                        checkout all projects as they were on a given date, in case of bitrot
  --num-shards NUM_SHARDS
                        number of shards to distribute projects across
  --shard-index SHARD_INDEX
                        run only on the given shard of projects

output:
  -o {full,diff,concise}, --output {full,diff,concise}
                        'full' shows both outputs + diff; 'diff' shows only the diff; 'concise'
                        shows only the diff but very compact
  --old-success         only output a result for a project if the old mypy run was successful

modes:
  --coverage            count files and lines covered
  --bisect              find first mypy revision to introduce a difference
  --bisect-output BISECT_OUTPUT
                        find first mypy revision with output matching given regex

primer:
  -j CONCURRENCY, --concurrency CONCURRENCY
                        number of subprocesses to use at a time
  --debug               print commands as they run
  --base-dir BASE_DIR   dir to store repos and venvs
  --clear               delete repos and venvs
```

## Examples

See the difference between HEAD and latest release with:
```
mypy_primer -o diff
```

See the impact of your risky change with:
```
mypy_primer --repo https://github.com/hauntsaninja/mypy.git --new my_risky_change --old master
```

See the impact of your risky typeshed change with:
```
mypy_primer --custom-typeshed-repo ~/dev/typeshed --new-typeshed my_risky_change --old-typeshed master --new v0.782 --old v0.782 -o concise
```

Filter to projects you care about:
```
mypy_primer -k hauntsaninja
```

Figure out what commit is causing a difference in the project you care about:
```
mypy_primer -k pandas --bisect
```

Figure out what commit is causing a specific error in the project you care about:
```
mypy_primer -k pandas --bisect-error 'Incompatible types in assignment'
```

Or on a local file / directory:
```
mypy_primer -p test.py --bisect --old v0.770
```

Find out what the hell mypy_primer is doing:
```
mypy_primer --debug
```

Or how much code it's covering (with your project selection):
```
mypy_primer --coverage -k pypa
```

For the record, the total is currently:
```
λ mypy_primer --coverage
Checking 94 projects...
Containing 19569 files...
Totalling to 5751015 lines...
```
(We use mypy internals to calculate this, so it's pretty accurate, if fragile)

## Contributing

I wrote this script up pretty quickly, so it's kind of hacky. Please improve it!

If you need it to do something different, it should be pretty easy to modify.

An easy thing to do is add more projects.

Some other things that could be done are:
- allow additional mypy flags
- add support for mypy plugins
- add bisection for typeshed
- make it possibe to dump structured output
- multiple mypy invocations for the same project
- improve CLI or output formatting
- ???

