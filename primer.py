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
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")
RevisionLike = Union[str, None, Callable[[Path], Awaitable[str]]]

BASE = Path("/tmp/mypy_primer")


class Style(str, Enum):
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


async def run(
    cmd: List[str], output: bool = False, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    if ARGS.debug:
        log = " ".join(cmd)
        log = f"{Style.BLUE}{log}"
        if "cwd" in kwargs:
            log += f"\t{Style.DIM} in {kwargs['cwd']}"
        log += Style.RESET
        print(log)

    if output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    else:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
    proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
    stdout_b, stderr_b = await proc.communicate()
    stdout = stdout_b.decode("utf-8") if stdout_b is not None else None
    stderr = stderr_b.decode("utf-8") if stderr_b is not None else None
    assert proc.returncode is not None
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


async def clone(repo_url: str, cwd: Path, shallow: bool = False) -> None:
    cmd = ["git", "clone", "--recurse-submodules", repo_url]
    if shallow:
        cmd += ["--depth", "1"]
    await run(cmd, cwd=cwd)


async def refresh(repo_dir: Path) -> None:
    await run(["git", "fetch"], cwd=repo_dir)
    await run(["git", "clean", "-ffxd"], cwd=repo_dir)
    await run(["git", "reset", "--hard", "origin/HEAD"], cwd=repo_dir)


async def get_revision_for_date(dt: date, repo_dir: Path) -> str:
    proc = await run(
        ["git", "rev-list", "-1", "--before", dt.isoformat(), "HEAD"], output=True, cwd=repo_dir
    )
    return proc.stdout.strip()


async def checkout(revision: str, repo_dir: Path) -> None:
    try:
        # try and interpret revision as an isoformatted date
        dt = date.fromisoformat(revision)
        revision = await get_revision_for_date(dt, repo_dir)
    except ValueError:
        pass
    await run(["git", "checkout", "--force", "--recurse-submodules", revision], cwd=repo_dir)


async def ensure_repo_at_revision(repo_url: str, cwd: Path, revision_like: RevisionLike) -> Path:
    repo_dir = cwd / Path(repo_url).stem
    if repo_dir.is_dir():
        await refresh(repo_dir)
    else:
        await clone(repo_url, cwd, shallow=revision_like is None)
    assert repo_dir.is_dir()

    # TODO: could fail if we had a shallow clone
    revision = (await revision_like(repo_dir)) if callable(revision_like) else revision_like
    if revision is not None:
        await checkout(revision, repo_dir)
    return repo_dir


async def get_recent_tag(repo_dir: Path) -> str:
    proc = await run(["git", "rev-list", "--tags", "-1"], output=True, cwd=repo_dir)
    proc = await run(["git", "describe", "--tags", proc.stdout.strip()], output=True, cwd=repo_dir)
    return proc.stdout.strip()


async def setup_mypy(mypy_dir: Path, revision_like: RevisionLike) -> Path:
    mypy_dir.mkdir(exist_ok=True)
    repo_dir = await ensure_repo_at_revision(ARGS.repo, mypy_dir, revision_like)

    venv.create(mypy_dir / "venv", with_pip=True, clear=True)
    await run([str(mypy_dir / "venv" / "bin" / "pip"), "install", str(repo_dir)])
    mypy_exe = mypy_dir / "venv" / "bin" / "mypy"

    assert mypy_exe.exists()
    return mypy_exe


async def setup_new_and_old_mypy(
    new_mypy_revision: Optional[str] = None, old_mypy_revision: Optional[str] = None
) -> Tuple[Path, Path]:
    async def get_old_revision(repo_dir: Path) -> str:
        if old_mypy_revision is None:
            return await get_recent_tag(repo_dir)
        return old_mypy_revision

    new_mypy, old_mypy = await asyncio.gather(
        setup_mypy(BASE / "new_mypy", new_mypy_revision),
        setup_mypy(BASE / "old_mypy", get_old_revision),
    )

    if ARGS.debug:
        new_mypy_version = (await run([str(new_mypy), "--version"], output=True)).stdout.strip()
        print(f"{Style.BLUE}new mypy version: {new_mypy_version}{Style.RESET}")
        old_mypy_version = (await run([str(old_mypy), "--version"], output=True)).stdout.strip()
        print(f"{Style.BLUE}old mypy version: {old_mypy_version}{Style.RESET}")

    return new_mypy, old_mypy


@dataclass
class Project:
    url: str
    mypy_cmd: str
    revision: Optional[str] = None
    deps: Optional[List[str]] = None
    # if failures_expected, there is no recent version of mypy which passes cleanly
    failures_expected: bool = True

    @property
    def name(self) -> str:
        return Path(self.url).stem

    def venv_dir(self, project_base: Path) -> Path:
        return project_base / f"_{self.name}_venv"

    async def setup(self, project_base: Path) -> None:
        repo_dir = await ensure_repo_at_revision(self.url, project_base, self.revision)
        assert repo_dir == project_base / self.name
        if self.deps:
            venv_dir = self.venv_dir(project_base)
            venv.create(venv_dir, with_pip=True, clear=True)
            for dep in self.deps:
                await run([str(venv_dir / "bin" / "pip"), "install", dep], cwd=repo_dir)

    async def run_mypy(self, mypy: str, project_base: Path) -> MypyResult:
        mypy_cmd = self.mypy_cmd
        assert "{mypy}" in mypy_cmd
        if self.deps:
            python_exe = self.venv_dir(project_base) / "bin" / "python"
            mypy_cmd += f" --python-executable={python_exe}"
        if ARGS.custom_typeshed_dir:
            mypy_cmd += f" --custom-typeshed-dir={ARGS.custom_typeshed_dir}"
        mypy_cmd += " --no-incremental --cache-dir=/dev/null"
        mypy_cmd = mypy_cmd.format(mypy=mypy)
        env = os.environ.copy()
        env["MYPY_FORCE_COLOR"] = "1"
        proc = await run(
            shlex.split(mypy_cmd),
            output=True,
            check=False,
            cwd=project_base / self.name,
            env=env,
        )
        if proc.returncode:
            status = "unexpected failure" if not self.failures_expected else "failure"
        else:
            status = "success"
        return MypyResult(mypy_cmd, status, proc.stdout)

    async def primer_result(self, new_mypy: str, old_mypy: str, project_base: Path) -> PrimerResult:
        await self.setup(project_base)
        # run them serially so that we are correctly limited by --concurrency
        new_result = await self.run_mypy(new_mypy, project_base)
        old_result = await self.run_mypy(old_mypy, project_base)
        return PrimerResult(self, new_result, old_result)


@dataclass
class MypyResult:
    command: str
    status: str
    output: str

    def __str__(self) -> str:
        ret = "> " + self.command + "\n"
        if self.status == "unexpected failure":
            ret += f"\t{Style.RED}{Style.BOLD}UNEXPECTED FAILURE{Style.RESET}\n"
        ret += textwrap.indent(self.output, "\t")
        return ret


@dataclass
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


async def yield_with_max_concurrency(
    streams: Iterator[Awaitable[T]], concurrency: int
) -> AsyncIterator[T]:
    assert isinstance(streams, Iterator)
    pending = set()

    def enqueue(c: int) -> None:
        try:
            for _ in range(c):
                pending.add(next(streams))
        except StopIteration:
            pass

    enqueue(concurrency)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)  # type: ignore
        concurrency = 0
        for fut in done:
            yield (await fut)
            concurrency += 1
        enqueue(concurrency)


async def primer() -> None:
    if BASE.exists() and ARGS.clear:
        shutil.rmtree(BASE)
    BASE.mkdir(exist_ok=True)

    new_mypy, old_mypy = await setup_new_and_old_mypy(
        new_mypy_revision=ARGS.new, old_mypy_revision=ARGS.old
    )

    project_base = BASE / "projects"
    project_base.mkdir(exist_ok=True)

    project_iter = (p for p in PROJECTS)
    if ARGS.project_selector:
        project_iter = (
            p for p in project_iter if re.search(ARGS.project_selector, p.url, flags=re.I)
        )
    if ARGS.success_expected:
        project_iter = (p for p in project_iter if not p.failures_expected)
    if ARGS.project_date:
        project_iter = (replace(p, revision=ARGS.project_date) for p in project_iter)

    results = (
        project.primer_result(str(new_mypy), str(old_mypy), project_base)
        for project in project_iter
    )

    concurrency = ARGS.concurrency
    if not concurrency:
        concurrency = multiprocessing.cpu_count()
    async for result in yield_with_max_concurrency(results, concurrency):
        print(result)


ARGS: argparse.Namespace


def main() -> None:
    parser = argparse.ArgumentParser()

    mypy_group = parser.add_argument_group("mypy")
    mypy_group.add_argument("--new", help="new mypy version, defaults to HEAD")
    mypy_group.add_argument("--old", help="old mypy version, defaults to latest tag")
    mypy_group.add_argument(
        "--repo", default="https://github.com/python/mypy.git", help="mypy repo to use"
    )
    mypy_group.add_argument("--custom-typeshed-dir", help="typeshed directory to use")

    proj_group = parser.add_argument_group("project filtration")
    proj_group.add_argument("-k", "--project-selector", help="regex to filter projects")
    proj_group.add_argument(
        "--success-expected",
        action="store_true",
        help="filter out projects where we could expect failures",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--diff-only", action="store_true", help="only output diff in mypy output"
    )

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument(
        "--project-date", help="checkout projects on a given date, in case of bitrot"
    )

    primer_group = parser.add_argument_group("primer")
    primer_group.add_argument(
        "-j", "--concurrency", default=0, type=int, help="number of subprocesses to use at a time"
    )
    primer_group.add_argument("--debug", action="store_true", help="print commands as they run")
    primer_group.add_argument(
        "--clear", action="store_true", help="delete previously used repos and venvs"
    )

    global ARGS
    ARGS = parser.parse_args(sys.argv[1:])

    asyncio.run(primer())


PROJECTS = [
    Project(
        url="https://github.com/python/mypy.git",
        mypy_cmd="{mypy} --config-file mypy_self_check.ini -p mypy -p mypyc",
        deps=["pytest"],
        failures_expected=False,
    ),
    Project(
        url="https://github.com/psf/black.git",
        mypy_cmd="{mypy} src",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/hauntsaninja/pyp.git",
        mypy_cmd="{mypy} --strict -m pyp",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/hauntsaninja/boostedblob.git",
        mypy_cmd="{mypy} boostedblob",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pytest-dev/pytest.git",
        mypy_cmd="{mypy} src testing",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pandas-dev/pandas.git",
        mypy_cmd="{mypy} pandas",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pycqa/pylint.git",
        mypy_cmd="{mypy} pylint/checkers --ignore-missing-imports",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/aio-libs/aiohttp.git",
        mypy_cmd="{mypy} aiohttp",
        deps=["-rrequirements/ci-wheel.txt"],
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pypa/bandersnatch.git",
        mypy_cmd="{mypy} src src/bandersnatch/tests",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/quora/asynq.git",
        mypy_cmd="{mypy} asynq",
        deps=["-rrequirements.txt"],
        failures_expected=False,
    ),
    Project(
        url="https://github.com/python-attrs/attrs.git",
        mypy_cmd=(
            "{mypy} src/attr/__init__.pyi src/attr/_version_info.pyi src/attr/converters.pyi"
            " src/attr/exceptions.pyi src/attr/filters.pyi src/attr/setters.pyi"
            " src/attr/validators.pyi tests/typing_example.py"
        ),
        failures_expected=False,
    ),
    Project(
        url="https://github.com/sphinx-doc/sphinx.git",
        mypy_cmd="{mypy} sphinx",
        deps=["docutils-stubs"],
        failures_expected=False,
    ),
    Project(
        url="https://github.com/scikit-learn/scikit-learn.git",
        mypy_cmd="{mypy} sklearn",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/scrapy/scrapy.git",
        mypy_cmd="{mypy} scrapy tests",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pypa/twine.git",
        mypy_cmd="{mypy} twine",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/more-itertools/more-itertools.git",
        mypy_cmd="{mypy} more_itertools",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pydata/xarray.git",
        mypy_cmd="{mypy} .",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/pallets/werkzeug.git",
        mypy_cmd="{mypy} src/werkzeug tests",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/bokeh/bokeh.git",
        mypy_cmd="{mypy} bokeh release",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/mystor/git-revise.git",
        mypy_cmd="{mypy} gitrevise",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/PyGithub/PyGithub.git",
        mypy_cmd="{mypy} github tests",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/we-like-parsers/pegen.git",
        mypy_cmd="{mypy}",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/zulip/zulip.git",
        mypy_cmd="{mypy} zerver zilencer zproject zthumbor tools analytics corporate scripts --platform=linux",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/dropbox/stone.git",
        mypy_cmd="{mypy} stone test",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/yelp/paasta.git",
        mypy_cmd="{mypy} paasta_tools",
        failures_expected=False,
    ),
    Project(
        url="https://github.com/PrefectHQ/prefect.git",
        mypy_cmd="{mypy} src",
        failures_expected=False,
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
