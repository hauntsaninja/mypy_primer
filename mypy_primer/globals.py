from __future__ import annotations

import argparse
import contextvars
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

import mypy_primer.utils


@dataclass
class _Args:
    # type checker group
    new: str | None
    old: str | None
    repo: str | None
    type_checker: str
    mypyc_compile_level: int | None

    custom_typeshed_repo: str
    new_typeshed: str | None
    old_typeshed: str | None
    new_prepend_path: Path | None
    old_prepend_path: Path | None

    additional_flags: list[str]

    # project group
    project_selector: str | None
    known_dependency_selector: str | None
    local_project: str | None
    expected_success: bool
    project_date: str | None

    shard_index: int | None
    num_shards: int | None

    # output group
    output: str
    old_success: bool
    show_speed_regression: bool

    # modes group
    coverage: bool
    bisect: bool
    bisect_output: str | None
    validate_expected_success: bool
    measure_project_runtimes: bool

    # primer group
    concurrency: int
    base_dir: Path
    debug: bool
    clear: bool

    projects_dir: Path = field(init=False)


ctx = contextvars.ContextVar[_Args]("args")


def parse_options(argv: list[str]) -> _Args:
    parser = argparse.ArgumentParser()

    type_checker_group = parser.add_argument_group("type checker")
    type_checker_group.add_argument(
        "--new",
        help=(
            "new type checker version, defaults to HEAD "
            "(pypi version, anything commit-ish, or isoformatted date)"
        ),
    )
    type_checker_group.add_argument(
        "--old",
        help=(
            "old type checker version, defaults to latest tag "
            "(pypi version, anything commit-ish, or isoformatted date)"
        ),
    )
    type_checker_group.add_argument(
        "--type-checker",
        default="mypy",
        choices=["mypy", "pyright", "knot", "pyrefly"],
        help="type checker to use",
    )
    type_checker_group.add_argument(
        "--repo",
        help=(
            "type checker repo to use (passed to git clone. if unspecified, we first try pypi, "
            "then fall back to github)"
        ),
    )
    type_checker_group.add_argument(
        "--mypyc-compile-level",
        default=None,
        type=int,
        help="Compile mypy with the given mypyc optimisation level",
    )

    type_checker_group.add_argument(
        "--custom-typeshed-repo",
        default="https://github.com/python/typeshed",
        help="typeshed repo to use (passed to git clone)",
    )
    type_checker_group.add_argument(
        "--new-typeshed",
        help="new typeshed version, defaults to vendored (commit-ish or isoformatted date)",
    )
    type_checker_group.add_argument(
        "--old-typeshed",
        help="old typeshed version, defaults to vendored (commit-ish, or isoformatted date)",
    )
    type_checker_group.add_argument(
        "--new-prepend-path",
        type=lambda s: Path(s).absolute(),
        help="a path to prepend to sys.path for new run",
    )
    type_checker_group.add_argument(
        "--old-prepend-path",
        type=lambda s: Path(s).absolute(),
        help="a path to prepend to sys.path for old run",
    )

    type_checker_group.add_argument(
        "--additional-flags",
        help="additional flags to pass to the type checker",
        nargs="*",
        default=[],
    )

    proj_group = parser.add_argument_group("project selection")
    proj_group.add_argument(
        "-k", "--project-selector", help="regex to filter projects (matches against location)"
    )
    proj_group.add_argument(
        "--known-dependency-selector",
        help="select all projects that depend on a given known project",
    )
    proj_group.add_argument(
        "-p",
        "--local-project",
        help=(
            "run only on the given file or directory. if a single file, supports a "
            "'# flags: ...' comment, like mypy unit tests"
        ),
    )
    proj_group.add_argument(
        "--expected-success",
        action="store_true",
        help=(
            "filter to hardcoded subset of projects marked as having had a recent mypy version succeed"
        ),
    )
    proj_group.add_argument(
        "--project-date",
        help="checkout all projects as they were on a given date, in case of bitrot",
    )
    proj_group.add_argument(
        "--num-shards", type=int, help="number of shards to distribute projects across"
    )
    proj_group.add_argument(
        "--shard-index", type=int, help="run only on the given shard of projects"
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "-o",
        "--output",
        choices=("full", "diff", "concise"),
        default="full",
        help=(
            "'full' shows both outputs + diff; 'diff' shows only the diff; 'concise' shows only"
            " the diff but very compact"
        ),
    )
    output_group.add_argument(
        "--old-success",
        action="store_true",
        help="only output a result for a project if the old type checker run was successful",
    )
    output_group.add_argument(
        "--show-speed-regression",
        action="store_true",
        help="show speed regression for each project",
    )

    modes_group = parser.add_argument_group("modes")
    modes_group.add_argument(
        "--coverage", action="store_true", help="count files and lines covered"
    )
    modes_group.add_argument(
        "--bisect", action="store_true", help="find first mypy revision to introduce a difference"
    )
    modes_group.add_argument(
        "--bisect-output", help="find first mypy revision with output matching given regex"
    )
    modes_group.add_argument(
        "--validate-expected-success",
        action="store_true",
        help="check if projects marked as expected success pass cleanly",
    )
    modes_group.add_argument(
        "--measure-project-runtimes", action="store_true", help="measure project runtimes"
    )

    primer_group = parser.add_argument_group("primer")
    primer_group.add_argument(
        "-j",
        "--concurrency",
        default=multiprocessing.cpu_count(),
        type=int,
        help="number of subprocesses to use at a time",
    )
    primer_group.add_argument("--debug", action="store_true", help="print commands as they run")
    primer_group.add_argument(
        "--base-dir",
        default=Path(mypy_primer.utils.TEMP_DIR) / "mypy_primer",
        type=Path,
        help="dir to store repos and venvs",
    )
    primer_group.add_argument("--clear", action="store_true", help="delete repos and venvs")

    ret = _Args(**vars(parser.parse_args(argv)))
    if (ret.num_shards is not None) != (ret.shard_index is not None):
        parser.error("--shard-index and --num-shards must be used together")
    return ret


def parse_options_and_set_ctx(argv: list[str]) -> _Args:
    args = parse_options(argv)
    ctx.set(args)
    return args
