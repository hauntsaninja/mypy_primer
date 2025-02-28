from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Awaitable, Iterator, TypeVar

from mypy_primer.git_utils import (
    RevisionLike,
    get_revision_for_revision_or_date,
    revision_or_recent_tag_fn,
)
from mypy_primer.globals import ctx, parse_options_and_set_ctx
from mypy_primer.model import Project, TypeCheckResult
from mypy_primer.projects import get_projects
from mypy_primer.type_checker import setup_mypy, setup_pyright, setup_typeshed
from mypy_primer.utils import Style, debug_print, line_count, run, strip_colour_code

T = TypeVar("T")


async def setup_new_and_old_mypy(
    new_mypy_revision: RevisionLike, old_mypy_revision: RevisionLike
) -> tuple[Path, Path]:
    new_mypy, old_mypy = await asyncio.gather(
        setup_mypy(
            ctx.get().base_dir / "new_mypy",
            new_mypy_revision,
            repo=ctx.get().repo,
            mypyc_compile_level=ctx.get().mypyc_compile_level,
        ),
        setup_mypy(
            ctx.get().base_dir / "old_mypy",
            old_mypy_revision,
            repo=ctx.get().repo,
            mypyc_compile_level=ctx.get().mypyc_compile_level,
        ),
    )

    if ctx.get().debug:
        (new_version, _), (old_version, _) = await asyncio.gather(
            run([str(new_mypy), "--version"], output=True),
            run([str(old_mypy), "--version"], output=True),
        )
        debug_print(f"{Style.BLUE}new mypy version: {new_version.stdout.strip()}{Style.RESET}")
        debug_print(f"{Style.BLUE}old mypy version: {old_version.stdout.strip()}{Style.RESET}")

    return new_mypy, old_mypy


async def setup_new_and_old_pyright(
    new_pyright_revision: RevisionLike, old_pyright_revision: RevisionLike
) -> tuple[Path, Path]:
    new_pyright, old_pyright = await asyncio.gather(
        setup_pyright(
            ctx.get().base_dir / "new_pyright",
            new_pyright_revision,
            repo=ctx.get().repo,
        ),
        setup_pyright(
            ctx.get().base_dir / "old_pyright",
            old_pyright_revision,
            repo=ctx.get().repo,
        ),
    )

    if ctx.get().debug:
        (new_version, _), (old_version, _) = await asyncio.gather(
            run([str(new_pyright), "--version"], output=True),
            run([str(old_pyright), "--version"], output=True),
        )
        debug_print(f"{Style.BLUE}new pyright version: {new_version.stdout.strip()}{Style.RESET}")
        debug_print(f"{Style.BLUE}old pyright version: {old_version.stdout.strip()}{Style.RESET}")

    return new_pyright, old_pyright


async def setup_new_and_old_typeshed(
    new_typeshed_revision: RevisionLike, old_typeshed_revision: RevisionLike
) -> tuple[Path | None, Path | None]:
    typeshed_repo = ctx.get().custom_typeshed_repo

    new_typeshed_dir = None
    old_typeshed_dir = None
    if ctx.get().new_typeshed:
        new_typeshed_dir = await setup_typeshed(
            ctx.get().base_dir / "new_typeshed",
            repo=typeshed_repo,
            revision_like=new_typeshed_revision,
        )
    if ctx.get().old_typeshed:
        old_typeshed_dir = await setup_typeshed(
            ctx.get().base_dir / "old_typeshed",
            repo=typeshed_repo,
            revision_like=old_typeshed_revision,
        )
    return new_typeshed_dir, old_typeshed_dir


# ==============================
# project utils
# ==============================


def select_projects() -> list[Project]:
    ARGS = ctx.get()
    if ARGS.local_project:
        return [Project.from_location(ARGS.local_project)]

    project_iter: Iterator[Project] = iter(
        p
        for p in get_projects()
        if not (p.min_python_version and sys.version_info < p.min_python_version)
    )
    if ARGS.type_checker == "pyright":
        project_iter = iter(p for p in project_iter if p.pyright_cmd is not None)
    if ARGS.project_selector:
        project_iter = iter(
            p for p in project_iter if re.search(ARGS.project_selector, p.location, flags=re.I)
        )
    if ARGS.known_dependency_selector:
        project_iter = iter(
            p for p in project_iter if ARGS.known_dependency_selector in (p.deps or [])
        )
    if ARGS.expected_success:
        project_iter = (p for p in project_iter if p.expected_success(ARGS.type_checker))
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

RECENT_MYPYS = ["1.10.1"]


async def validate_expected_success() -> None:
    """Check correctness of hardcoded Project.expected_success"""
    ARGS = ctx.get()

    assert ARGS.type_checker == "mypy"

    recent_mypy_exes = await asyncio.gather(
        *[
            setup_mypy(
                ARGS.base_dir / ("mypy_" + recent_mypy),
                recent_mypy,
                repo=ARGS.repo,
                mypyc_compile_level=ARGS.mypyc_compile_level,
            )
            for recent_mypy in RECENT_MYPYS
        ]
    )

    async def inner(project: Project) -> str | None:
        await project.setup()
        success = None
        for mypy_exe in recent_mypy_exes:
            mypy_result = await project.run_mypy(mypy_exe, typeshed_dir=None, prepend_path=None)
            if ARGS.debug:
                debug_print(format(Style.BLUE))
                debug_print(mypy_result)
                debug_print(format(Style.RESET))
            if mypy_result.success:
                success = mypy_exe
                break
        if bool(success) and not project.expected_mypy_success:
            return (
                f"Project {project.location} succeeded with {success}, "
                "but is not marked as expecting success"
            )
        if not bool(success) and project.expected_mypy_success:
            return f"Project {project.location} did not succeed, but is marked as expecting success"
        return None

    results = await asyncio.gather(*[inner(project) for project in select_projects()])
    for result in results:
        if result:
            print(result)


async def measure_project_runtimes() -> None:
    """Check type checker runtime over each project."""
    ARGS = ctx.get()

    if ARGS.type_checker == "mypy":
        base_name = "timer_mypy" if ARGS.new is None else f"timer_mypy_{ARGS.new}"
        type_checker_exe = await setup_mypy(
            ARGS.base_dir / base_name,
            ARGS.new or RECENT_MYPYS[0],
            repo=ARGS.repo,
            mypyc_compile_level=ARGS.mypyc_compile_level,
        )
    elif ARGS.type_checker == "pyright":
        type_checker_exe = await setup_pyright(
            ARGS.base_dir / "timer_pyright",
            ARGS.new,
            repo=ARGS.repo,
        )
    else:
        raise ValueError(f"Unknown type checker {ARGS.type_checker}")

    async def inner(project: Project) -> tuple[float, Project]:
        await project.setup()
        result = await project.run_typechecker(
            type_checker_exe, typeshed_dir=None, prepend_path=None
        )
        return (result.runtime, project)

    projects = select_projects()
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
async def bisect() -> None:
    ARGS = ctx.get()

    assert not ARGS.new_typeshed
    assert not ARGS.old_typeshed

    if ARGS.type_checker == "mypy":
        type_checker_exe = await setup_mypy(
            ARGS.base_dir / "bisect_mypy",
            revision_or_recent_tag_fn(ARGS.old),
            repo=ARGS.repo,
            mypyc_compile_level=ARGS.mypyc_compile_level,
            editable=True,
        )
        repo_dir = ARGS.base_dir / "bisect_mypy" / "mypy"
    elif ARGS.type_checker == "pyright":
        type_checker_exe = await setup_pyright(
            ARGS.base_dir / "bisect_pyright",
            revision_or_recent_tag_fn(ARGS.old),
            repo=ARGS.repo,
        )
        repo_dir = ARGS.base_dir / "bisect_pyright" / "pyright"
    else:
        raise ValueError(f"Unknown type checker {ARGS.type_checker}")

    assert repo_dir.is_dir()

    projects = select_projects()
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
            await run(["npm", "run", "install:all"], cwd=repo_dir)
            await run(["npm", "run", "build"], cwd=repo_dir / "packages" / "pyright")

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


async def coverage() -> None:
    ARGS = ctx.get()
    assert ARGS.type_checker == "mypy"
    mypy_exe = await setup_mypy(
        ARGS.base_dir / "new_mypy",
        revision_like=ARGS.new,
        repo=ARGS.repo,
        mypyc_compile_level=ARGS.mypyc_compile_level,
    )

    projects = select_projects()
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


async def primer() -> int:
    projects = select_projects()
    ARGS = ctx.get()

    if ARGS.type_checker == "mypy":
        new_type_checker, old_type_checker = await setup_new_and_old_mypy(
            new_mypy_revision=ARGS.new,
            old_mypy_revision=revision_or_recent_tag_fn(ARGS.old),
        )
    elif ARGS.type_checker == "pyright":
        new_type_checker, old_type_checker = await setup_new_and_old_pyright(
            new_pyright_revision=ARGS.new,
            old_pyright_revision=revision_or_recent_tag_fn(ARGS.old),
        )
    else:
        raise ValueError(f"Unknown type checker {ARGS.type_checker}")

    new_typeshed_dir, old_typeshed_dir = await setup_new_and_old_typeshed(
        ARGS.new_typeshed, ARGS.old_typeshed
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
            coro = coverage()
        elif ARGS.bisect or ARGS.bisect_output:
            coro = bisect()
        elif ARGS.validate_expected_success:
            coro = validate_expected_success()
        elif ARGS.measure_project_runtimes:
            coro = measure_project_runtimes()
        else:
            coro = primer()

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
