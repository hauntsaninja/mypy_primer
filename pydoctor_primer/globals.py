from __future__ import annotations

import argparse
import contextvars
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

import pydoctor_primer.utils


@dataclass
class _Args:
    # type checker group
    new: str | None
    old: str | None
    repo: str | None

    additional_flags: list[str]

    # project group
    project_selector: str | None
    local_project: str | None
    expected_success: bool
    project_date: str | None

    shard_index: int | None
    num_shards: int | None

    # output group
    output: str
    old_success: bool

    # modes group
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

    type_checker_group = parser.add_argument_group("pydoctor")
    type_checker_group.add_argument(
        "--new",
        help=(
            "new pydoctor version, defaults to HEAD "
            "(pypi version, anything commit-ish, or isoformatted date)"
        ),
    )
    type_checker_group.add_argument(
        "--old",
        help=(
            "old pydoctor version, defaults to latest tag "
            "(pypi version, anything commit-ish, or isoformatted date)"
        ),
    )
    type_checker_group.add_argument(
        "--repo",
        help=(
            "pydoctor repo to use (passed to git clone. if unspecified, we first try pypi, "
            "then fall back to github)"
        ),
    )

    type_checker_group.add_argument(
        "--additional-flags",
        help="additional flags to pass to pydoctor",
        nargs="*",
        default=[],
    )

    proj_group = parser.add_argument_group("project selection")
    proj_group.add_argument(
        "-k", "--project-selector", help="regex to filter projects (matches against location)"
    )
    proj_group.add_argument(
        "-p",
        "--local-project",
        help=(
            "run pydoctor only on the given file or directory"
        ),
    )
    proj_group.add_argument(
        "--expected-success",
        action="store_true",
        help=(
            "filter to hardcoded subset of projects where some recent pydoctor "
            "version succeeded. also look at: --old-success"
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
        help="only output a result for a project if the old pydoctor run was successful",
    )

    modes_group = parser.add_argument_group("modes")
    modes_group.add_argument(
        "--bisect", action="store_true", help="find first pydoctor revision to introduce a difference"
    )
    modes_group.add_argument(
        "--bisect-output", help="find first pydoctor revision with output matching given regex"
    )
    modes_group.add_argument(
        "--validate-expected-success", action="store_true", help=argparse.SUPPRESS
    )
    modes_group.add_argument(
        "--measure-project-runtimes", action="store_true", help=argparse.SUPPRESS
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
        default=Path(pydoctor_primer.utils.TEMP_DIR) / "pydoctor_primer",
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
