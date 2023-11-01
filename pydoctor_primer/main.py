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

from pydoctor_primer.git_utils import (
    RevisionLike,
    get_revision_for_revision_or_date,
    revision_or_recent_tag_fn,
)
from pydoctor_primer.globals import ctx, parse_options_and_set_ctx
from pydoctor_primer.model import Project, TypeCheckResult
from pydoctor_primer.projects import get_projects
from pydoctor_primer.type_checker import setup_pydoctor
from pydoctor_primer.utils import Style, debug_print, line_count, run, strip_colour_code

T = TypeVar("T")


async def setup_new_and_old_pydoctor(
    new_revision: RevisionLike, old_revision: RevisionLike
) -> tuple[Path, Path]:
    new_exe, old_exe = await asyncio.gather(
        setup_pydoctor(
            ctx.get().base_dir / "new_pydoctor",
            new_revision,
            repo=ctx.get().repo,
        ),
        setup_pydoctor(
            ctx.get().base_dir / "old_pydoctor",
            old_revision,
            repo=ctx.get().repo,
        ),
    )

    if ctx.get().debug:
        (new_version, _), (old_version, _) = await asyncio.gather(
            run([str(new_exe), "--version"], output=True),
            run([str(old_exe), "--version"], output=True),
        )
        debug_print(f"{Style.BLUE}new: {new_version.stdout.strip()}{Style.RESET}")
        debug_print(f"{Style.BLUE}old: {old_version.stdout.strip()}{Style.RESET}")

    return new_exe, old_exe

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
    if ARGS.project_selector:
        project_iter = iter(
            p for p in project_iter if re.search(ARGS.project_selector, p.location, flags=re.I)
        )
    if ARGS.expected_success:
        project_iter = (p for p in project_iter if p.expected_success)
    if ARGS.project_date:
        project_iter = (replace(p, revision=ARGS.project_date) for p in project_iter)

    projects = list(project_iter)
    if projects == []:
        raise ValueError("No projects selected!")

    if ARGS.num_shards:
        assert ARGS.shard_index is not None
        shard_costs = [0] * ARGS.num_shards
        shard_projects: list[list[Project]] = [[] for _ in range(ARGS.num_shards)]
        for p in sorted(projects, key=lambda p: (1, p.location), reverse=True):
            min_shard = min(range(ARGS.num_shards), key=lambda i: shard_costs[i])
            shard_costs[min_shard] += 1
            shard_projects[min_shard].append(p)
        return shard_projects[ARGS.shard_index]
    return projects


# ==============================
# hidden entrypoint logic
# ==============================

RECENT_PYDOCTOR = ["23.9.0", "23.4.1", "22.9.1", "22.7.0"]


async def validate_expected_success() -> None:
    """Check correctness of hardcoded Project.expected_success"""
    ARGS = ctx.get()

    recent_pydoctor_exes = await asyncio.gather(
        *[
            setup_pydoctor(
                ARGS.base_dir / ("pydoctor_" + pydoctor_dir),
                pydoctor_dir,
                repo=ARGS.repo,
            )
            for pydoctor_dir in RECENT_PYDOCTOR
        ]
    )

    async def inner(project: Project) -> str | None:
        await project.setup()
        success = None
        for exe in recent_pydoctor_exes:
            result = await project.run_pydoctor(exe)
            if ARGS.debug:
                debug_print(format(Style.BLUE))
                debug_print(result)
                debug_print(format(Style.RESET))
            if result.success:
                success = exe
                break
        if bool(success) and not project.expected_success:
            return (
                f"Project {project.location} succeeded with {success}, "
                "but is not marked as expecting success"
            )
        if not bool(success) and project.expected_success:
            return f"Project {project.location} did not succeed, but is marked as expecting success"
        return None

    results = await asyncio.gather(*[inner(project) for project in select_projects()])
    for result in results:
        if result:
            print(result)


async def measure_project_runtimes() -> None:
    """Check pydoctor's runtime over each project."""
    ARGS = ctx.get()
    
    exe = await setup_pydoctor(
        ARGS.base_dir / "runtimes",
        ARGS.new or RECENT_PYDOCTOR[0],
        repo=ARGS.repo,
    )

    async def inner(project: Project) -> tuple[float, Project]:
        await project.setup()
        result = await project.run_pydoctor(exe)
        return (result.runtime, project)

    results = sorted(
        (await asyncio.gather(*[inner(project) for project in select_projects()])), reverse=True
    )
    for time_taken, project in results:
        print(f"{time_taken:6.2f}  {project.location}")


# ==============================
# entrypoint logic
# ==============================


# TODO: can't bisect over typeshed commits yet
async def bisect() -> None:
    ARGS = ctx.get()

    exe = await setup_pydoctor(
        ARGS.base_dir / "bisect",
        revision_or_recent_tag_fn(ARGS.old),
        repo=ARGS.repo,
        editable=True,
    )
    repo_dir = ARGS.base_dir / "bisect" / "pydoctor"
    assert repo_dir.is_dir()

    projects = select_projects()
    await asyncio.wait([project.setup() for project in projects])

    async def run_wrapper(project: Project) -> tuple[str, TypeCheckResult]:
        return project.name, (await project.run_pydoctor(str(exe)))

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


async def primer() -> int:
    projects = select_projects()
    ARGS = ctx.get()

    new_type_checker, old_type_checker = await setup_new_and_old_pydoctor(
        ARGS.new,
        revision_or_recent_tag_fn(ARGS.old),
    )

    results = [
        project.primer_result(
            new_type_checker=str(new_type_checker),
            old_type_checker=str(old_type_checker),
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
        if ARGS.bisect or ARGS.bisect_output:
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
