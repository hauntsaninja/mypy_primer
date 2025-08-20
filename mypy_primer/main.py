from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, TypeVar

from mypy_primer.git_utils import (
    RevisionLike,
    get_revision_for_revision_or_date,
    revision_or_recent_tag_fn,
)
from mypy_primer.globals import _Args, parse_options_and_set_ctx
from mypy_primer.model import Project, TypeCheckResult
from mypy_primer.projects import get_projects
from mypy_primer.type_checker import (
    setup_mypy,
    setup_pyrefly,
    setup_pyright,
    setup_ty,
    setup_typeshed,
)
from mypy_primer.utils import Style, debug_print, get_npm, line_count, run, strip_colour_code

T = TypeVar("T")


def setup_type_checker(
    ARGS: _Args,
    *,
    revision_like: RevisionLike,
    suffix: str,
    typeshed_dir: Path | None,
) -> Awaitable[Path]:
    setup_fn: Callable[..., Awaitable[Path]]
    kwargs: dict[str, Any]

    if ARGS.type_checker == "mypy":
        setup_fn = setup_mypy
        kwargs = {"repo": ARGS.repo, "mypyc_compile_level": ARGS.mypyc_compile_level}
    elif ARGS.type_checker == "pyright":
        setup_fn = setup_pyright
        kwargs = {"repo": ARGS.repo}
    elif ARGS.type_checker == "ty":
        setup_fn = setup_ty
        kwargs = {"repo": ARGS.repo}
    elif ARGS.type_checker == "pyrefly":
        setup_fn = setup_pyrefly
        kwargs = {"repo": ARGS.repo, "typeshed_dir": typeshed_dir}
    else:
        raise ValueError(f"Unknown type checker {ARGS.type_checker}")

    return setup_fn(
        ARGS.base_dir / f"{ARGS.type_checker}_{suffix}", revision_like=revision_like, **kwargs
    )


async def setup_new_and_old_type_checker(
    ARGS: _Args,
    new_typeshed_dir: Path | None,
    old_typeshed_dir: Path | None,
) -> tuple[Path, Path]:
    new_revision = ARGS.new
    old_revision = revision_or_recent_tag_fn(ARGS.old)

    new_exe, old_exe = await asyncio.gather(
        setup_type_checker(
            ARGS,
            revision_like=new_revision,
            suffix="new",
            typeshed_dir=new_typeshed_dir,
        ),
        setup_type_checker(
            ARGS,
            revision_like=old_revision,
            suffix="old",
            typeshed_dir=old_typeshed_dir,
        ),
    )

    if ARGS.debug:
        (new_version, _), (old_version, _) = await asyncio.gather(
            run([str(new_exe), "--version"], output=True),
            run([str(old_exe), "--version"], output=True),
        )
        debug_print(
            f"{Style.BLUE}new {ARGS.type_checker} version: {new_version.stdout.strip()}{Style.RESET}"
        )
        debug_print(
            f"{Style.BLUE}old {ARGS.type_checker} version: {old_version.stdout.strip()}{Style.RESET}"
        )

    return new_exe, old_exe


async def setup_new_and_old_typeshed(ARGS: _Args) -> tuple[Path | None, Path | None]:
    typeshed_repo = ARGS.custom_typeshed_repo
    new_typeshed_revision = ARGS.new_typeshed
    old_typeshed_revision = ARGS.old_typeshed

    new_typeshed_dir = None
    old_typeshed_dir = None
    if ARGS.new_typeshed:
        new_typeshed_dir = await setup_typeshed(
            ARGS.base_dir / "new_typeshed", repo=typeshed_repo, revision_like=new_typeshed_revision
        )
    if ARGS.old_typeshed:
        old_typeshed_dir = await setup_typeshed(
            ARGS.base_dir / "old_typeshed", repo=typeshed_repo, revision_like=old_typeshed_revision
        )
    return new_typeshed_dir, old_typeshed_dir


# ==============================
# project utils
# ==============================


def select_projects(ARGS: _Args) -> list[Project]:
    if ARGS.local_project:
        return [Project.from_location(ARGS.local_project)]

    project_iter: Iterator[Project] = iter(
        p
        for p in get_projects()
        if not (p.min_python_version and sys.version_info < p.min_python_version)
    )

    if ARGS.type_checker == "mypy":
        project_iter = iter(p for p in project_iter if p.mypy_cmd is not None)
    if ARGS.type_checker == "pyright":
        project_iter = iter(p for p in project_iter if p.pyright_cmd is not None)
    # if ARGS.type_checker == "ty":
    #     project_iter = iter(p for p in project_iter if p.ty_cmd is not None)

    if ARGS.project_selector:
        project_iter = iter(
            p for p in project_iter if re.search(ARGS.project_selector, p.location, flags=re.I)
        )
    if ARGS.known_dependency_selector:
        project_iter = iter(
            p for p in project_iter if ARGS.known_dependency_selector in (p.deps or [])
        )
    if ARGS.expected_success:
        project_iter = (p for p in project_iter if ARGS.type_checker in p.expected_success)
    if ARGS.project_date:
        project_iter = (replace(p, revision=ARGS.project_date) for p in project_iter)

    projects = [
        p
        for p in project_iter
        if p.supported_platforms is None or sys.platform in p.supported_platforms
    ]
    if projects == []:
        raise ValueError("No projects selected!")

    if ARGS.num_shards:
        assert ARGS.shard_index is not None
        shard_costs = [0] * ARGS.num_shards
        shard_projects: list[list[Project]] = [[] for _ in range(ARGS.num_shards)]
        for p in sorted(
            projects,
            key=lambda p: (p.cost_for_type_checker(ARGS.type_checker), p.location),
            reverse=True,
        ):
            min_shard = min(range(ARGS.num_shards), key=lambda i: shard_costs[i])
            shard_costs[min_shard] += p.cost_for_type_checker(ARGS.type_checker)
            shard_projects[min_shard].append(p)
        return shard_projects[ARGS.shard_index]
    return projects


# ==============================
# hidden entrypoint logic
# ==============================

RECENT_VERSIONS = {"mypy": ["1.17.1"], "pyright": ["1.1.399"]}


async def validate_expected_success(ARGS: _Args) -> None:
    """Check correctness of hardcoded Project.expected_success"""

    recent_type_checker_exes = await asyncio.gather(
        *[
            setup_type_checker(
                ARGS,
                revision_like=recent_type_checker,
                suffix=recent_type_checker,
                typeshed_dir=None,
            )
            for recent_type_checker in RECENT_VERSIONS[ARGS.type_checker]
        ]
    )

    async def inner(project: Project) -> str | None:
        await project.setup()
        success = None
        for type_checker_exe in recent_type_checker_exes:
            result = await project.run_typechecker(
                type_checker_exe, typeshed_dir=None, prepend_path=None
            )
            if ARGS.debug:
                debug_print(result)
            if result.success:
                success = type_checker_exe
                break

        expected_success = ARGS.type_checker in project.expected_success
        if bool(success) and not expected_success:
            return (
                f"Project {project.location} succeeded with {success}, "
                "but is not marked as expecting success"
            )
        if not bool(success) and expected_success:
            return f"Project {project.location} did not succeed, but is marked as expecting success"
        return None

    results = await asyncio.gather(*[inner(project) for project in select_projects(ARGS)])
    for result in results:
        if result:
            print(result)


async def measure_project_runtimes(ARGS: _Args) -> None:
    """Check type checker runtime over each project."""
    type_checker_exe = await setup_type_checker(
        ARGS,
        revision_like=ARGS.new or RECENT_VERSIONS[ARGS.type_checker][0],
        suffix="timer_" + (ARGS.new if ARGS.new else ""),
        typeshed_dir=None,
    )

    async def inner(project: Project) -> tuple[float, Project]:
        await project.setup()
        result = await project.run_typechecker(
            type_checker_exe, typeshed_dir=None, prepend_path=None
        )
        return (result.runtime, project)

    projects = select_projects(ARGS)
    results = []
    for fut in asyncio.as_completed([inner(project) for project in projects]):
        time_taken, project = await fut
        results.append((time_taken, project))
        print(f"[{len(results)}/{len(projects)}] {time_taken:6.2f}s  {project.location}")

    results.sort(reverse=True)
    print("\n" * 5)
    print("Results:")
    for time_taken, project in results:
        print(f"{time_taken:6.2f}s  {project.location}")


# ==============================
# entrypoint logic
# ==============================


# TODO: can't bisect over typeshed commits yet
async def bisect(ARGS: _Args) -> None:
    assert not ARGS.new_typeshed
    assert not ARGS.old_typeshed

    if ARGS.type_checker == "mypy":
        type_checker_exe = await setup_mypy(
            ARGS.base_dir / "bisect_mypy",
            revision_like=revision_or_recent_tag_fn(ARGS.old),
            repo=ARGS.repo,
            mypyc_compile_level=ARGS.mypyc_compile_level,
            editable=True,  # important
        )
        repo_dir = ARGS.base_dir / "bisect_mypy" / "mypy"
    elif ARGS.type_checker == "pyright":
        type_checker_exe = await setup_pyright(
            ARGS.base_dir / "bisect_pyright",
            revision_like=revision_or_recent_tag_fn(ARGS.old),
            repo=ARGS.repo,
        )
        repo_dir = ARGS.base_dir / "bisect_pyright" / "pyright"
    else:
        raise ValueError(f"Unknown type checker {ARGS.type_checker}")

    assert repo_dir.is_dir()

    projects = select_projects(ARGS)
    await asyncio.gather(*[project.setup() for project in projects])

    async def run_wrapper(project: Project) -> tuple[str, TypeCheckResult]:
        return project.name, (
            await project.run_typechecker(type_checker_exe, typeshed_dir=None, prepend_path=None)
        )

    results_fut = await asyncio.gather(*(run_wrapper(project) for project in projects))
    old_results: dict[str, TypeCheckResult] = dict(results_fut)
    if ARGS.debug:
        debug_print("\n".join(str(result) for result in old_results.values()))
        debug_print(format(Style.RESET))

    # Note git bisect start will clean up old bisection state
    await run(["git", "bisect", "start"], cwd=repo_dir, output=True)
    await run(["git", "bisect", "good"], cwd=repo_dir, output=True)
    new_revision = await get_revision_for_revision_or_date(ARGS.new or "origin/HEAD", repo_dir)
    await run(["git", "bisect", "bad", new_revision], cwd=repo_dir, output=True)

    def are_results_good(results: dict[str, TypeCheckResult]) -> bool:
        if ARGS.bisect_output:
            return not any(
                re.search(ARGS.bisect_output, strip_colour_code(results[project.name].output))
                for project in projects
            )
        return all(
            results[project.name].output == old_results[project.name].output for project in projects
        )

    assert are_results_good(old_results)

    while True:
        await run(["git", "submodule", "update", "--init"], cwd=repo_dir)

        if ARGS.type_checker == "pyright":
            npm = get_npm()
            await run([npm, "run", "install:all"], cwd=repo_dir)
            await run([npm, "run", "build"], cwd=repo_dir / "packages" / "pyright")

        results_fut = await asyncio.gather(*(run_wrapper(project) for project in projects))
        results: dict[str, TypeCheckResult] = dict(results_fut)

        state = "good" if are_results_good(results) else "bad"
        proc, _ = await run(["git", "bisect", state], output=True, cwd=repo_dir)

        if "first bad commit" in proc.stdout:
            print(proc.stdout)
            return

        if ARGS.debug:
            debug_print("\n".join(str(result) for result in results.values()))
            debug_print(proc.stdout)
            debug_print(format(Style.RESET))


async def coverage(ARGS: _Args) -> None:
    assert ARGS.type_checker == "mypy"

    mypy_exe = await setup_type_checker(
        ARGS,
        revision_like=ARGS.new,
        suffix="new",
        typeshed_dir=None,
    )

    projects = select_projects(ARGS)
    if sys.platform == "win32":
        mypy_python = mypy_exe.parent / "python.exe"
    else:
        mypy_python = mypy_exe.parent / "python"

    assert mypy_python.exists()

    all_paths = await asyncio.gather(
        *[project.mypy_source_paths(str(mypy_python)) for project in projects]
    )

    project_to_paths: dict[str, int] = {}
    project_to_lines: dict[str, int] = {}
    for project, paths in zip(projects, all_paths):
        project_to_paths[project.location] = len(paths)
        project_to_lines[project.location] = sum(map(line_count, paths))

    for project in sorted(projects, key=lambda p: project_to_lines[p.location], reverse=True):
        p = project.location
        print(p, project_to_lines[p], project_to_paths[p])

    print(f"Checking {len(projects)} projects...")
    print(f"Containing {sum(project_to_paths.values())} files...")
    print(f"Totalling to {sum(project_to_lines.values())} lines...")


async def primer(ARGS: _Args) -> int:
    projects = select_projects(ARGS)

    new_typeshed_dir, old_typeshed_dir = await setup_new_and_old_typeshed(ARGS)
    new_type_checker, old_type_checker = await setup_new_and_old_type_checker(
        ARGS,
        new_typeshed_dir=new_typeshed_dir,
        old_typeshed_dir=old_typeshed_dir,
    )

    results = [
        project.primer_result(
            new_type_checker=new_type_checker,
            old_type_checker=old_type_checker,
            new_typeshed=new_typeshed_dir,
            old_typeshed=old_typeshed_dir,
            new_prepend_path=ARGS.new_prepend_path,
            old_prepend_path=ARGS.old_prepend_path,
        )
        for project in projects
    ]
    retcode = 0
    for result_fut in asyncio.as_completed(results):
        result = await result_fut
        if ARGS.old_success and not result.old_result.success:
            continue
        if ARGS.output == "full":
            print(result.format_full())
        elif ARGS.output == "diff":
            print(result.format_diff_only())
        elif ARGS.output == "concise":
            # using ARGS.output == "concise" also causes us to:
            # - always pass in --no-pretty and --no-error-summary
            concise = result.format_concise()
            if concise:
                print(concise)
                print()
        if not retcode and result.diff:
            retcode = 1
    return retcode


def main() -> None:
    if sys.platform == "win32":
        # Enables ANSI escape characters in terminal without resorting to ctypes or colorama
        os.system("")

    def inner() -> int | None:
        ARGS = parse_options_and_set_ctx(sys.argv[1:])

        if ARGS.base_dir.exists() and ARGS.clear:
            shutil.rmtree(ARGS.base_dir)
        ARGS.base_dir = ARGS.base_dir.absolute()
        ARGS.base_dir.mkdir(exist_ok=True)
        ARGS.projects_dir = ARGS.base_dir / "projects"
        ARGS.projects_dir.mkdir(exist_ok=True)

        coro: Awaitable[int | None]
        if ARGS.coverage:
            coro = coverage(ARGS)
        elif ARGS.bisect or ARGS.bisect_output:
            coro = bisect(ARGS)
        elif ARGS.validate_expected_success:
            coro = validate_expected_success(ARGS)
        elif ARGS.measure_project_runtimes:
            coro = measure_project_runtimes(ARGS)
        else:
            coro = primer(ARGS)

        try:
            retcode = asyncio.run(coro)
        finally:
            if ARGS.base_dir.exists() and ARGS.clear:
                shutil.rmtree(ARGS.base_dir)
        return retcode

    try:
        retcode = inner()
    except Exception:
        traceback.print_exc()
        retcode = 70
    sys.exit(retcode)
