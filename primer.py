from __future__ import annotations

import argparse
import asyncio
import difflib
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import venv
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")
RevisionLike = Union[str, None, Callable[[Path], Awaitable[str]]]


# ==============================
# utils
# ==============================


class Style(str, Enum):
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def strip_colour_code(text: str) -> str:
    return re.sub("\\x1b.*?m", "", text)


_semaphore: Optional[asyncio.Semaphore] = None


async def run(
    cmd: List[str], output: bool = False, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    if output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    else:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)

    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.BoundedSemaphore(ARGS.concurrency)
    async with _semaphore:
        if ARGS.debug:
            log = " ".join(map(shlex.quote, cmd))
            log = f"{Style.BLUE}{log}"
            if "cwd" in kwargs:
                log += f"\t{Style.DIM} in {kwargs['cwd']}"
            log += Style.RESET
            print(log)

        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        stdout_b, stderr_b = await proc.communicate()

    stdout = stdout_b.decode("utf-8") if stdout_b is not None else None
    stderr = stderr_b.decode("utf-8") if stderr_b is not None else None
    assert proc.returncode is not None
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def line_count(path: Path) -> int:
    buf_size = 1024 * 1024
    with open(path, "rb") as f:
        buf_iter = iter(lambda: f.raw.read(buf_size), b"")  # type: ignore
        return sum(buf.count(b"\n") for buf in buf_iter)


# ==============================
# git utils
# ==============================


async def clone(repo_url: str, cwd: Path, shallow: bool = False) -> None:
    cmd = ["git", "clone", "--recurse-submodules", repo_url]
    if shallow:
        cmd += ["--depth", "1"]
    await run(cmd, cwd=cwd)


async def refresh(repo_dir: Path) -> None:
    await run(["git", "fetch"], cwd=repo_dir)
    await run(["git", "clean", "-ffxd"], cwd=repo_dir)
    await run(["git", "reset", "--hard", "origin/HEAD", "--recurse-submodules"], cwd=repo_dir)


async def checkout(revision: str, repo_dir: Path) -> None:
    await run(["git", "checkout", "--force", "--recurse-submodules", revision], cwd=repo_dir)


async def get_revision_for_date(dt: date, repo_dir: Path) -> str:
    proc = await run(
        ["git", "rev-list", "-1", "--before", dt.isoformat(), "HEAD"], output=True, cwd=repo_dir
    )
    return proc.stdout.strip()


async def get_revision_for_revision_or_date(revision: str, repo_dir: Path) -> str:
    try:
        # try and interpret revision as an isoformatted date
        dt = date.fromisoformat(revision)
        return await get_revision_for_date(dt, repo_dir)
    except ValueError:
        return revision


async def ensure_repo_at_revision(repo_url: str, cwd: Path, revision_like: RevisionLike) -> Path:
    repo_dir = cwd / Path(repo_url).stem
    if repo_dir.is_dir():
        await refresh(repo_dir)
    else:
        await clone(repo_url, cwd, shallow=revision_like is None)
    assert repo_dir.is_dir()

    for retry in (True, False):
        try:
            revision = (await revision_like(repo_dir)) if callable(revision_like) else revision_like
            if revision is not None:
                revision = await get_revision_for_revision_or_date(revision, repo_dir)
                await checkout(revision, repo_dir)
        except subprocess.CalledProcessError:
            if retry:
                await run(["git", "fetch", "--unshallow"], cwd=repo_dir)
                continue
            raise
        break
    return repo_dir


async def get_recent_tag(repo_dir: Path) -> str:
    proc = await run(["git", "rev-list", "--tags", "-1"], output=True, cwd=repo_dir)
    proc = await run(["git", "describe", "--tags", proc.stdout.strip()], output=True, cwd=repo_dir)
    return proc.stdout.strip()


def revision_or_recent_tag_fn(revision: Optional[str]) -> RevisionLike:
    return revision if revision is not None else get_recent_tag


# ==============================
# mypy utils
# ==============================


async def setup_mypy(mypy_dir: Path, revision_like: RevisionLike, editable: bool = False) -> Path:
    mypy_dir.mkdir(exist_ok=True)
    repo_dir = await ensure_repo_at_revision(ARGS.repo, mypy_dir, revision_like)

    venv_dir = mypy_dir / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    install_cmd = [str(venv_dir / "bin" / "pip"), "install"]
    if editable:
        install_cmd.append("--editable")
    install_cmd.append(str(repo_dir))
    await run(install_cmd)

    mypy_exe = venv_dir / "bin" / "mypy"
    assert mypy_exe.exists()
    return mypy_exe


async def setup_new_and_old_mypy(
    new_mypy_revision: RevisionLike, old_mypy_revision: RevisionLike
) -> Tuple[Path, Path]:
    new_mypy, old_mypy = await asyncio.gather(
        setup_mypy(ARGS.base_dir / "new_mypy", new_mypy_revision),
        setup_mypy(ARGS.base_dir / "old_mypy", old_mypy_revision),
    )

    if ARGS.debug:
        new_version, old_version = await asyncio.gather(
            run([str(new_mypy), "--version"], output=True),
            run([str(old_mypy), "--version"], output=True),
        )
        print(f"{Style.BLUE}new mypy version: {new_version.stdout.strip()}{Style.RESET}")
        print(f"{Style.BLUE}old mypy version: {old_version.stdout.strip()}{Style.RESET}")

    return new_mypy, old_mypy


# ==============================
# classes
# ==============================


@dataclass(frozen=True)
class Project:
    url: str
    mypy_cmd: str
    revision: Optional[str] = None
    deps: Optional[List[str]] = None
    # if expected_success, there is a recent version of mypy which passes cleanly
    expected_success: bool = False

    @property
    def name(self) -> str:
        return Path(self.url).stem

    @property
    def venv_dir(self) -> Path:
        return ARGS.projects_dir / f"_{self.name}_venv"

    async def setup(self) -> None:
        repo_dir = await ensure_repo_at_revision(self.url, ARGS.projects_dir, self.revision)
        assert repo_dir == ARGS.projects_dir / self.name
        if self.deps:
            venv.create(self.venv_dir, with_pip=True, clear=True)
            for dep in self.deps:
                await run([str(self.venv_dir / "bin" / "pip"), "install", dep], cwd=repo_dir)

    def get_mypy_cmd(self, mypy: str, additional_flags: Sequence[str] = ()) -> str:
        mypy_cmd = self.mypy_cmd
        assert mypy_cmd.startswith("{mypy}")
        if self.deps:
            python_exe = self.venv_dir / "bin" / "python"
            mypy_cmd += f" --python-executable={python_exe}"
        if additional_flags:
            mypy_cmd += " " + " ".join(additional_flags)
        mypy_cmd += " --no-incremental --cache-dir=/dev/null"
        mypy_cmd = mypy_cmd.format(mypy=mypy)
        return mypy_cmd

    async def run_mypy(self, mypy: str, additional_flags: Sequence[str] = ()) -> MypyResult:
        mypy_cmd = self.get_mypy_cmd(mypy, additional_flags)
        env = os.environ.copy()
        env["MYPY_FORCE_COLOR"] = "1"
        proc = await run(
            shlex.split(mypy_cmd),
            output=True,
            check=False,
            cwd=ARGS.projects_dir / self.name,
            env=env,
        )
        return MypyResult(
            mypy_cmd, proc.stderr + proc.stdout, not bool(proc.returncode), self.expected_success
        )

    async def primer_result(self, new_mypy: str, old_mypy: str) -> PrimerResult:
        await self.setup()
        new_additional_flags = []
        if ARGS.new_custom_typeshed_dir:
            new_additional_flags = [f"--custom-typeshed-dir={ARGS.new_custom_typeshed_dir}"]
        new_result, old_result = await asyncio.gather(
            self.run_mypy(new_mypy, new_additional_flags), self.run_mypy(old_mypy)
        )
        return PrimerResult(self, new_result, old_result)

    async def source_paths(self, mypy_python: str) -> List[Path]:
        await self.setup()
        mypy_cmd = self.get_mypy_cmd("mypy")
        program = f"""
import io, mypy.fscache, mypy.main
args = "{mypy_cmd}".split()[1:]
fscache = mypy.fscache.FileSystemCache()
sources, _ = mypy.main.process_options(args, io.StringIO(), io.StringIO(), fscache=fscache)
for source in sources:
    if source.path is not None:  # can happen for modules...
        print(source.path)
"""
        proc = await run(
            [mypy_python, "-c", program], output=True, cwd=ARGS.projects_dir / self.name
        )
        return [ARGS.projects_dir / self.name / p for p in proc.stdout.splitlines()]


@dataclass(frozen=True)
class MypyResult:
    command: str
    output: str
    success: bool
    expected_success: bool

    def __str__(self) -> str:
        ret = "> " + self.command + "\n"
        if self.expected_success and not self.success:
            ret += f"\t{Style.RED}{Style.BOLD}UNEXPECTED FAILURE{Style.RESET}\n"
        ret += textwrap.indent(self.output, "\t")
        return ret


@dataclass(frozen=True)
class PrimerResult:
    project: Project
    new_result: MypyResult
    old_result: MypyResult

    def diff(self) -> str:
        d = difflib.Differ()
        diff = d.compare(self.old_result.output.splitlines(), self.new_result.output.splitlines())
        return "\n".join(line for line in diff if line[0] in ("+", "-"))

    def __str__(self) -> str:
        ret = f"\n{Style.BOLD}{self.project.name}{Style.RESET}\n"
        ret += self.project.url + "\n"

        if not ARGS.diff_only:
            ret += "----------\n\n"
            ret += "old mypy\n"
            ret += str(self.old_result)
            ret += "----------\n\n"
            ret += "new mypy\n"
            ret += str(self.new_result)

        diff = self.diff()
        if diff:
            ret += "----------\n\n"
            ret += "diff\n"
            ret += textwrap.indent(diff, "\t")
            ret += "\n"
        ret += "==========\n\n"
        return ret


# ==============================
# main logic
# ==============================


def select_projects() -> Iterator[Project]:
    project_iter = (p for p in PROJECTS)
    if ARGS.project_selector:
        project_iter = (
            p for p in project_iter if re.search(ARGS.project_selector, p.url, flags=re.I)
        )
    if ARGS.expected_success:
        project_iter = (p for p in project_iter if p.expected_success)
    if ARGS.project_date:
        project_iter = (replace(p, revision=ARGS.project_date) for p in project_iter)
    return project_iter


async def bisect() -> None:
    mypy_exe = await setup_mypy(
        ARGS.base_dir / "bisect_mypy", revision_or_recent_tag_fn(ARGS.old), editable=True
    )
    repo_dir = ARGS.base_dir / "bisect_mypy" / "mypy"
    assert repo_dir.is_dir()

    projects = list(select_projects())
    await asyncio.wait([project.setup() for project in projects])

    async def run_wrapper(project: Project) -> Tuple[Project, MypyResult]:
        return project, (await project.run_mypy(str(mypy_exe), ARGS.new_custom_typeshed_dir))

    results_fut = await asyncio.gather(*(run_wrapper(project) for project in projects))
    old_results: Dict[Project, MypyResult] = dict(results_fut)

    await run(["git", "bisect", "reset"], cwd=repo_dir)
    await run(["git", "bisect", "start"], cwd=repo_dir)
    await run(["git", "bisect", "good"], cwd=repo_dir)
    new_revision = await get_revision_for_revision_or_date(ARGS.new or "origin/HEAD", repo_dir)
    await run(["git", "bisect", "bad", new_revision], cwd=repo_dir)

    def are_results_good(results: Dict[Project, MypyResult]) -> bool:
        if ARGS.bisect_error:
            return not any(
                re.search(ARGS.bisect_error, strip_colour_code(results[project].output))
                for project in projects
            )
        return all(results[project].output == old_results[project].output for project in projects)

    assert are_results_good(old_results)

    while True:
        await run(["git", "submodule", "update", "--init"], cwd=repo_dir)
        results_fut = await asyncio.gather(*(run_wrapper(project) for project in projects))
        results: Dict[Project, MypyResult] = dict(results_fut)

        state = "good" if are_results_good(results) else "bad"
        proc = await run(["git", "bisect", state], output=True, cwd=repo_dir)

        if ARGS.debug:
            print(f"{Style.BLUE}{proc.stdout}{Style.RESET}")

        if "first bad commit" in proc.stdout:
            print(proc.stdout)
            return


async def coverage() -> None:
    mypy_exe = await setup_mypy(ARGS.base_dir / "new_mypy", ARGS.new)

    projects = list(select_projects())
    mypy_python = mypy_exe.parent / "python"
    assert mypy_python.exists()
    all_paths = [
        path
        for paths in asyncio.as_completed(
            [project.source_paths(str(mypy_python)) for project in projects]
        )
        for path in (await paths)
    ]
    num_lines = sum(map(line_count, all_paths))

    print(f"Checking {len(projects)} projects...")
    print(f"Containing {len(all_paths)} files...")
    print(f"Totalling to {num_lines} lines...")


async def primer() -> None:
    new_mypy, old_mypy = await setup_new_and_old_mypy(
        new_mypy_revision=ARGS.new, old_mypy_revision=revision_or_recent_tag_fn(ARGS.old)
    )

    results = [project.primer_result(str(new_mypy), str(old_mypy)) for project in select_projects()]
    for result_fut in asyncio.as_completed(results):
        result = await result_fut
        if ARGS.old_success and not result.old_result.success:
            continue
        print(result)


def parse_options(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    mypy_group = parser.add_argument_group("mypy")
    mypy_group.add_argument("--new", help="new mypy version, defaults to HEAD")
    mypy_group.add_argument("--old", help="old mypy version, defaults to latest tag")
    mypy_group.add_argument(
        "--repo", default="https://github.com/python/mypy.git", help="mypy repo to use"
    )
    mypy_group.add_argument(
        "--new-custom-typeshed-dir", help="typeshed directory to use with new mypy"
    )

    proj_group = parser.add_argument_group("project selection")
    proj_group.add_argument("-k", "--project-selector", help="regex to filter projects")
    proj_group.add_argument(
        "--expected-success",
        action="store_true",
        help="filter to projects where a recent mypy version succeeded",
    )
    proj_group.add_argument(
        "--project-date", help="checkout projects on a given date, in case of bitrot"
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--old-success", action="store_true", help="only output if old mypy run was successful"
    )
    output_group.add_argument(
        "--diff-only", action="store_true", help="only output the diff between mypy runs"
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
        default=Path("/tmp/mypy_primer"),
        type=Path,
        help="dir to store repos and venvs",
    )
    primer_group.add_argument(
        "--clear", action="store_true", help="delete previously used repos and venvs"
    )
    primer_group.add_argument(
        "--coverage", action="store_true", help="find files and lines covered"
    )
    primer_group.add_argument("--bisect", action="store_true", help="find bad mypy revision")
    primer_group.add_argument(
        "--bisect-error", help="find bad mypy revision based on an error regex"
    )

    return parser.parse_args(argv)


ARGS: argparse.Namespace


def main() -> None:
    global ARGS
    ARGS = parse_options(sys.argv[1:])

    if ARGS.base_dir.exists() and ARGS.clear:
        shutil.rmtree(ARGS.base_dir)
    ARGS.base_dir.mkdir(exist_ok=True)
    ARGS.projects_dir = ARGS.base_dir / "projects"
    ARGS.projects_dir.mkdir(exist_ok=True)

    if ARGS.coverage:
        coro = coverage()
    elif ARGS.bisect or ARGS.bisect_error:
        coro = bisect()
    else:
        coro = primer()

    try:
        asyncio.run(coro)
    finally:
        if ARGS.base_dir.exists() and ARGS.clear:
            shutil.rmtree(ARGS.base_dir)


# ==============================
# project definitions
# ==============================


PROJECTS = [
    Project(
        url="https://github.com/python/mypy.git",
        mypy_cmd="{mypy} --config-file mypy_self_check.ini -p mypy -p mypyc",
        deps=["pytest"],
        expected_success=True,
    ),
    Project(
        url="https://github.com/psf/black.git",
        mypy_cmd="{mypy} src",
        expected_success=True,
    ),
    Project(
        url="https://github.com/hauntsaninja/pyp.git",
        mypy_cmd="{mypy} --strict -m pyp",
        expected_success=True,
    ),
    Project(
        url="https://github.com/hauntsaninja/boostedblob.git",
        mypy_cmd="{mypy} boostedblob",
        expected_success=True,
    ),
    Project(
        url="https://github.com/pytest-dev/pytest.git",
        mypy_cmd="{mypy} src testing",
        expected_success=True,
    ),
    Project(
        url="https://github.com/pandas-dev/pandas.git",
        mypy_cmd="{mypy} pandas",
        expected_success=True,
    ),
    Project(
        url="https://github.com/pycqa/pylint.git",
        mypy_cmd="{mypy} pylint/checkers --ignore-missing-imports",
        expected_success=True,
    ),
    Project(
        url="https://github.com/aio-libs/aiohttp.git",
        mypy_cmd="{mypy} aiohttp",
        deps=["-rrequirements/ci-wheel.txt"],
        expected_success=True,
    ),
    Project(
        url="https://github.com/pypa/bandersnatch.git",
        mypy_cmd="{mypy} src src/bandersnatch/tests",
        expected_success=True,
    ),
    Project(
        url="https://github.com/quora/asynq.git",
        mypy_cmd="{mypy} asynq",
        deps=["-rrequirements.txt"],
        expected_success=True,
    ),
    Project(
        url="https://github.com/python-attrs/attrs.git",
        mypy_cmd=(
            "{mypy} src/attr/__init__.pyi src/attr/_version_info.pyi src/attr/converters.pyi"
            " src/attr/exceptions.pyi src/attr/filters.pyi src/attr/setters.pyi"
            " src/attr/validators.pyi tests/typing_example.py"
        ),
        expected_success=True,
    ),
    Project(
        url="https://github.com/sphinx-doc/sphinx.git",
        mypy_cmd="{mypy} sphinx",
        deps=["docutils-stubs"],
        expected_success=True,
    ),
    Project(
        url="https://github.com/scikit-learn/scikit-learn.git",
        mypy_cmd="{mypy} sklearn",
        expected_success=True,
    ),
    Project(
        url="https://github.com/scrapy/scrapy.git",
        mypy_cmd="{mypy} scrapy tests",
        expected_success=True,
    ),
    Project(
        url="https://github.com/pypa/twine.git",
        mypy_cmd="{mypy} twine",
        expected_success=True,
    ),
    Project(
        url="https://github.com/more-itertools/more-itertools.git",
        mypy_cmd="{mypy} more_itertools",
        expected_success=True,
    ),
    Project(
        url="https://github.com/pydata/xarray.git",
        mypy_cmd="{mypy} .",
        expected_success=True,
    ),
    Project(
        url="https://github.com/pallets/werkzeug.git",
        mypy_cmd="{mypy} src/werkzeug tests",
        expected_success=True,
    ),
    Project(
        url="https://github.com/bokeh/bokeh.git",
        mypy_cmd="{mypy} bokeh release",
        expected_success=True,
    ),
    Project(
        url="https://github.com/mystor/git-revise.git",
        mypy_cmd="{mypy} gitrevise",
        expected_success=True,
    ),
    Project(
        url="https://github.com/PyGithub/PyGithub.git",
        mypy_cmd="{mypy} github tests",
        expected_success=True,
    ),
    Project(
        url="https://github.com/we-like-parsers/pegen.git",
        mypy_cmd="{mypy}",
        expected_success=True,
    ),
    Project(
        url="https://github.com/zulip/zulip.git",
        mypy_cmd=(
            "{mypy} zerver zilencer zproject zthumbor tools analytics corporate scripts"
            " --platform=linux"
        ),
        expected_success=True,
    ),
    Project(
        url="https://github.com/dropbox/stone.git",
        mypy_cmd="{mypy} stone test",
        expected_success=True,
    ),
    Project(
        url="https://github.com/yelp/paasta.git",
        mypy_cmd="{mypy} paasta_tools",
        expected_success=True,
    ),
    Project(
        url="https://github.com/PrefectHQ/prefect.git",
        mypy_cmd="{mypy} src",
        expected_success=True,
    ),
    # failures expected...
    Project(
        url="https://github.com/pyppeteer/pyppeteer.git",
        mypy_cmd="{mypy} pyppeteer --config-file tox.ini",
        deps=["."],
    ),
    Project(
        url="https://github.com/pypa/pip.git",
        mypy_cmd="{mypy} src",
    ),
    Project(
        # relies on setup.py to create a version.py file
        url="https://github.com/pytorch/vision.git",
        mypy_cmd="{mypy}",
    ),
    # TODO: needs mypy-zope
    # Project(
    #     url="https://github.com/twisted/twisted.git",
    #     mypy_cmd="{mypy} src",
    # ),
    Project(
        url="https://github.com/tornadoweb/tornado.git",
        mypy_cmd="{mypy} tornado",
    ),
    Project(
        url="https://github.com/sympy/sympy.git",
        mypy_cmd="{mypy} sympy",
    ),
    Project(
        url="https://github.com/scipy/scipy.git",
        mypy_cmd="{mypy} scipy",
        deps=["git+git://github.com/numpy/numpy-stubs.git@master"],
    ),
]


if __name__ == "__main__":
    main()
