from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import venv
from collections import defaultdict
from dataclasses import dataclass, field, replace
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


def debug_print(obj: Any) -> None:
    assert ARGS.debug
    print(obj, file=sys.stderr)


def stable_hash(p: Project) -> int:
    return int(hashlib.md5(p.location.encode("utf-8")).hexdigest(), 16)


_semaphore: Optional[asyncio.Semaphore] = None


async def run(
    cmd: Union[str, List[str]],
    *,
    shell: bool = False,
    output: bool = False,
    check: bool = True,
    **kwargs: Any,
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
            log = cmd if shell else " ".join(map(shlex.quote, cmd))
            log = f"{Style.BLUE}{log}"
            if "cwd" in kwargs:
                log += f"\t{Style.DIM} in {kwargs['cwd']}"
            log += Style.RESET
            debug_print(log)

        if shell:
            assert isinstance(cmd, str)
            proc = await asyncio.create_subprocess_shell(cmd, **kwargs)
        else:
            assert isinstance(cmd, list)
            proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        stdout_b, stderr_b = await proc.communicate()

    stdout = stdout_b.decode("utf-8") if stdout_b is not None else None
    stderr = stderr_b.decode("utf-8") if stderr_b is not None else None
    assert proc.returncode is not None
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def line_count(path: Path) -> int:
    if path.is_dir():
        return 0
    buf_size = 1024 * 1024
    with open(path, "rb") as f:
        buf_iter = iter(lambda: f.raw.read(buf_size), b"")
        return sum(buf.count(b"\n") for buf in buf_iter)  # type: ignore


# ==============================
# git utils
# ==============================


async def clone(repo_url: str, cwd: Path, shallow: bool = False) -> None:
    if os.path.exists(repo_url):
        repo_url = os.path.abspath(repo_url)
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
    repo_dir = cwd / Path(repo_url).name
    if repo_dir.is_dir():
        await refresh(repo_dir)
    else:
        await clone(repo_url, cwd, shallow=revision_like is None)
    assert repo_dir.is_dir()

    if revision_like is None:
        return repo_dir
    revision = (await revision_like(repo_dir)) if callable(revision_like) else revision_like
    revision = await get_revision_for_revision_or_date(revision, repo_dir)

    for retry in (True, False):
        # checking out a local branch is probably not what we want, so preemptively delete
        try:
            await checkout("origin/HEAD", repo_dir)
            await run(["git", "branch", "-D", revision], output=True, cwd=repo_dir)
        except subprocess.CalledProcessError as e:
            # out of caution, be defensive about the error here
            if "not found" not in e.stderr:
                raise

        try:
            await checkout(revision, repo_dir)
            break
        except subprocess.CalledProcessError:
            # assume checkout failed due to having a shallow clone. try to unshallow our clone
            # and then retry
            if retry:
                refspec = "+refs/heads/*:refs/remotes/origin/*"
                await run(["git", "config", "remote.origin.fetch", refspec], cwd=repo_dir)
                await run(
                    ["git", "fetch", "--unshallow", "--all", "--tags"], cwd=repo_dir, check=False
                )
                continue
            raise
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
    venv_dir = mypy_dir / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    return await setup_mypy_into_venv(mypy_dir, venv_dir, revision_like, editable)


async def setup_mypy_into_venv(
    mypy_dir: Path, venv_dir: Path, revision_like: RevisionLike, editable: bool
) -> Path:
    pip_exe = str(venv_dir / "bin" / "pip")

    if ARGS.mypyc_compile_level is not None:
        editable = True

    install_from_repo = True
    if (
        isinstance(revision_like, str)
        and not editable
        and ARGS.repo == "https://github.com/python/mypy"
    ):
        # optimistically attempt to install the revision of mypy we want from pypi
        try:
            await run([pip_exe, "install", f"mypy=={revision_like}"])
            install_from_repo = False
        except subprocess.CalledProcessError:
            install_from_repo = True

    if install_from_repo:
        repo_dir = await ensure_repo_at_revision(ARGS.repo, mypy_dir, revision_like)
        if ARGS.mypyc_compile_level is not None:
            env = os.environ.copy()
            env["MYPYC_OPT_LEVEL"] = str(ARGS.mypyc_compile_level)
            python_exe = str(venv_dir / "bin" / "python")
            await run([pip_exe, "install", "typing_extensions", "mypy_extensions"])
            await run(
                [python_exe, "setup.py", "--use-mypyc", "build_ext", "--inplace"],
                cwd=repo_dir,
                env=env,
            )
        install_cmd = [pip_exe, "install"]
        if editable:
            install_cmd.append("--editable")
        install_cmd.append(str(repo_dir))
        install_cmd.append("tomli")
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
        debug_print(f"{Style.BLUE}new mypy version: {new_version.stdout.strip()}{Style.RESET}")
        debug_print(f"{Style.BLUE}old mypy version: {old_version.stdout.strip()}{Style.RESET}")

    return new_mypy, old_mypy


async def setup_new_and_old_typeshed(
    new_typeshed_revision: RevisionLike, old_typeshed_revision: RevisionLike
) -> Tuple[Optional[Path], Optional[Path]]:
    typeshed_repo = ARGS.custom_typeshed_repo

    new_typeshed_dir = None
    old_typeshed_dir = None
    if ARGS.new_typeshed:
        parent_dir = ARGS.base_dir / "new_typeshed"
        if parent_dir.exists():
            shutil.rmtree(parent_dir)
        parent_dir.mkdir(exist_ok=True)
        new_typeshed_dir = await ensure_repo_at_revision(
            typeshed_repo, ARGS.base_dir / "new_typeshed", new_typeshed_revision
        )
    if ARGS.old_typeshed:
        parent_dir = ARGS.base_dir / "old_typeshed"
        if parent_dir.exists():
            shutil.rmtree(parent_dir)
        parent_dir.mkdir(exist_ok=True)
        old_typeshed_dir = await ensure_repo_at_revision(
            typeshed_repo, parent_dir, old_typeshed_revision
        )
    return new_typeshed_dir, old_typeshed_dir


# ==============================
# classes
# ==============================


@dataclass(frozen=True)
class Project:
    location: str
    mypy_cmd: str
    revision: Optional[str] = None
    pip_cmd: Optional[str] = None
    # if expected_success, there is a recent version of mypy which passes cleanly
    expected_success: bool = False
    mypyc_cmd: Optional[str] = None

    @property
    def name(self) -> str:
        return Path(self.location).name

    @property
    def venv_dir(self) -> Path:
        return ARGS.projects_dir / f"_{self.name}_venv"  # type: ignore[no-any-return]

    async def setup(self) -> None:
        if Path(self.location).exists():
            repo_dir = ARGS.projects_dir / self.name
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            if Path(self.location).is_file():
                # allow a project to be a single file
                repo_dir.mkdir()
                shutil.copy(Path(self.location), repo_dir / Path(self.location).name)
            elif Path(self.location).is_dir() and not (Path(self.location) / ".git").exists():
                # allow a project to be a local folder (that isn't a git repo)
                shutil.copytree(Path(self.location), repo_dir)
        else:
            # usually projects are something clonable
            repo_dir = await ensure_repo_at_revision(
                self.location, ARGS.projects_dir, self.revision
            )
        assert repo_dir == ARGS.projects_dir / self.name
        if self.pip_cmd:
            assert "{pip}" in self.pip_cmd
            venv.create(self.venv_dir, with_pip=True, clear=True)
            await run(
                self.pip_cmd.format(pip=str(self.venv_dir / "bin" / "pip")),
                shell=True,
                cwd=repo_dir,
            )

    def get_mypy_cmd(self, mypy: Union[str, Path], additional_flags: Sequence[str] = ()) -> str:
        mypy_cmd = self.mypy_cmd
        assert "{mypy}" in self.mypy_cmd
        if self.pip_cmd:
            python_exe = self.venv_dir / "bin" / "python"
            mypy_cmd += f" --python-executable={python_exe}"
        if additional_flags:
            mypy_cmd += " " + " ".join(additional_flags)
        if ARGS.output == "concise":
            mypy_cmd += "  --no-pretty --no-error-summary"
        mypy_cmd += " --no-incremental --cache-dir=/dev/null --show-traceback"
        mypy_cmd += " --soft-error-limit ' -1'"
        mypy_cmd = mypy_cmd.format(mypy=mypy)
        return mypy_cmd

    async def run_mypy(self, mypy: Union[str, Path], typeshed_dir: Optional[Path]) -> MypyResult:
        additional_flags = []
        env = os.environ.copy()
        env["MYPY_FORCE_COLOR"] = "1"

        if typeshed_dir is not None:
            additional_flags.append(f"--custom-typeshed-dir={typeshed_dir}")
            add_to_mypypath = ":".join(map(str, typeshed_dir.glob("stubs/*")))
            if "MYPYPATH" in env:
                env["MYPYPATH"] += ":" + add_to_mypypath
            else:
                env["MYPYPATH"] = add_to_mypypath

        mypy_cmd = self.get_mypy_cmd(mypy, additional_flags)
        proc = await run(
            mypy_cmd,
            shell=True,
            output=True,
            check=False,
            cwd=ARGS.projects_dir / self.name,
            env=env,
        )

        output = proc.stderr + proc.stdout
        if typeshed_dir is not None:
            # Differing line numbers and typeshed paths create noisy diffs.
            # Not a problem for stdlib because mypy silences errors from --custom-typeshed-dir.
            output = "".join(
                line
                for line in output.splitlines(keepends=True)
                if not line.startswith(str(typeshed_dir / "stubs"))
            )
        return MypyResult(mypy_cmd, output, not bool(proc.returncode), self.expected_success)

    async def run_mypyc(self, python: str) -> MypyResult:
        assert "{python}" in self.mypyc_cmd
        mypyc_cmd = self.mypyc_cmd.format(python=python)
        proc = await run(
            mypyc_cmd, shell=True, output=True, check=False, cwd=ARGS.projects_dir / self.name
        )
        output = proc.stderr + proc.stdout
        return MypyResult(mypyc_cmd, output, not bool(proc.returncode), self.expected_success)

    async def primer_result(
        self,
        new_mypy: str,
        new_mypy_revision: RevisionLike,
        old_mypy: str,
        old_mypy_revision: RevisionLike,
        new_typeshed: Optional[Path],
        old_typeshed: Optional[Path],
    ) -> PrimerResult:
        await self.setup()
        if self.mypyc_cmd:
            await setup_mypy_into_venv(
                ARGS.base_dir / "new_mypy", self.venv_dir, new_mypy_revision, editable=False
            )
            new_mypyc_result = await self.run_mypyc(python=self.venv_dir / "bin" / "python")
            await self.setup()
            await setup_mypy_into_venv(
                ARGS.base_dir / "old_mypy", self.venv_dir, old_mypy_revision, editable=False
            )
            old_mypyc_result = await self.run_mypyc(python=self.venv_dir / "bin" / "python")
        else:
            new_mypyc_result, old_mypyc_result = None, None

        new_result, old_result = await asyncio.gather(
            self.run_mypy(new_mypy, new_typeshed),
            self.run_mypy(old_mypy, old_typeshed),
        )
        return PrimerResult(self, new_result, new_mypyc_result, old_result, old_mypyc_result)

    async def source_paths(self, mypy_python: str) -> List[Path]:
        await self.setup()
        mypy_cmd = self.get_mypy_cmd(mypy="mypyprimersentinel")
        mypy_cmd = mypy_cmd.split("mypyprimersentinel", maxsplit=1)[1]
        program = """
import io, sys, mypy.fscache, mypy.main
args = sys.argv[1:]
fscache = mypy.fscache.FileSystemCache()
sources, _ = mypy.main.process_options(args, io.StringIO(), io.StringIO(), fscache=fscache)
for source in sources:
    if source.path is not None:  # can happen for modules...
        print(source.path)
"""
        # the extra shell stuff here makes sure we expand globs in mypy_cmd
        proc = await run(
            f"{mypy_python} -c {shlex.quote(program)} {mypy_cmd}",
            output=True,
            cwd=ARGS.projects_dir / self.name,
            shell=True,
        )
        return [ARGS.projects_dir / self.name / p for p in proc.stdout.splitlines()]

    @classmethod
    def from_location(cls, location: str) -> Project:
        additional_flags = ""
        if Path(location).is_file():
            with open(location) as f:
                header = f.readline().strip()
                if header.startswith("# flags:"):
                    additional_flags = header[len("# flags:") :]
        return Project(
            location=location,
            mypy_cmd=f"{{mypy}} {location} {additional_flags}",
        )


@dataclass(frozen=True)
class MypyResult:
    command: str
    output: str
    success: bool
    expected_success: bool

    def __str__(self) -> str:
        ret = "> " + self.command + "\n"
        if self.expected_success and not self.success:
            ret += f"{Style.RED}{Style.BOLD}UNEXPECTED FAILURE{Style.RESET}\n"
        ret += textwrap.indent(self.output, "\t")
        return ret


@dataclass(frozen=True)
class PrimerResult:
    project: Project
    new_result: MypyResult
    new_mypyc_result: Optional[MypyResult]
    old_result: MypyResult
    old_mypyc_result: Optional[MypyResult]
    diff: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diff", self._get_diff())

    def _get_diff(self) -> str:
        d = difflib.Differ()

        old_output = self.old_result.output
        new_output = self.new_result.output

        if "error: INTERNAL ERROR" in old_output:
            old_output = re.sub('File ".*/mypy', 'File "', old_output)
        if "error: INTERNAL ERROR" in new_output:
            new_output = re.sub('File ".*/mypy', 'File "', new_output)

        old_lines = old_output.splitlines()
        new_lines = new_output.splitlines()
        # Hide "note" lines which contain ARGS.base_dir... this hides differences between
        # file paths, e.g., when mypy points to a stub definition.
        old_lines = [line for line in old_lines if not re.search(f"{ARGS.base_dir}.*: note:", line)]
        new_lines = [line for line in new_lines if not re.search(f"{ARGS.base_dir}.*: note:", line)]
        diff = d.compare(old_lines, new_lines)
        diff_lines = [line for line in diff if line[0] in ("+", "-")]

        # mypy's output appears to be nondeterministic for some same line errors, e.g. on pypa/pip
        # Work around that by ignoring identical removal and addition pairs, e.g.
        # "- a.py:1: error xyz" and "+ a.py:1: error xyz"
        net_change: Dict[str, int] = defaultdict(int)
        for line in diff_lines:
            net_change[line[2:]] += 1 if line[0] == "+" else -1

        output_lines = []
        for line in diff_lines:
            if line[0] == "+" and net_change[line[2:]] > 0:
                output_lines.append(line)
                net_change[line[2:]] -= 1
            elif line[0] == "-" and net_change[line[2:]] < 0:
                output_lines.append(line)
                net_change[line[2:]] += 1

        return "\n".join(output_lines)

    def header(self) -> str:
        ret = f"\n{Style.BOLD}{self.project.name}{Style.RESET}\n"
        ret += self.project.location + "\n"
        mypyc_result = self.mypyc_result(verbose=True)
        if mypyc_result:
            ret += f"{mypyc_result}\n"
        return ret

    def mypyc_result(self, *, verbose: bool = False) -> str:
        mypyc_result = ""
        if self.old_mypyc_result and self.new_mypyc_result:
            mypyc_result = "[mypyc "
            if not self.new_mypyc_result.success:
                if self.old_mypyc_result.success:
                    mypyc_result += f"🔥{Style.RED}{Style.BOLD}BROKE{Style.RESET}]"
                    if verbose:
                        mypyc_result += f"\n{self.new_mypyc_result.output}"
                else:
                    mypyc_result += "❌]"
                    if verbose:
                        mypyc_result += f"\n{self.new_mypyc_result.output}"
            else:
                mypyc_result += "✅]"
        return mypyc_result

    def format_concise(self) -> str:
        mypyc_result = self.mypyc_result(verbose=False)
        if self.diff or mypyc_result:
            return f"{self.project.name} ({self.project.location}) {mypyc_result}\n{self.diff}"
        return ""

    def format_diff_only(self) -> str:
        ret = self.header()

        if self.diff:
            ret += "----------\n"
            ret += textwrap.indent(self.diff, "\t")
            ret += "\n"

        ret += "==========\n"
        return ret

    def format_full(self) -> str:
        ret = self.header()
        ret += "----------\n\n"
        ret += "old mypy\n"
        ret += str(self.old_result)
        ret += "----------\n\n"
        ret += "new mypy\n"
        ret += str(self.new_result)

        if self.diff:
            ret += "----------\n\n"
            ret += "diff\n"
            ret += textwrap.indent(self.diff, "\t")
            ret += "\n"

        ret += "==========\n"
        return ret


# ==============================
# project utils
# ==============================


def select_projects() -> Iterator[Project]:
    if ARGS.local_project:
        return iter([Project.from_location(ARGS.local_project)])
    project_iter: Iterator[Project] = iter(p for p in PROJECTS)
    if ARGS.project_selector:
        projects = [
            p for p in project_iter if re.search(ARGS.project_selector, p.location, flags=re.I)
        ]
        if projects == []:
            raise Exception(f"No projects were selected by -k {ARGS.project_selector}")
        project_iter = iter(projects)
    if ARGS.expected_success:
        project_iter = (p for p in project_iter if p.expected_success)
    if ARGS.project_date:
        project_iter = (replace(p, revision=ARGS.project_date) for p in project_iter)
    if ARGS.num_shards:
        project_iter = (
            p for p in project_iter if stable_hash(p) % ARGS.num_shards == ARGS.shard_index
        )
    return project_iter


# ==============================
# hidden entrypoint logic
# ==============================

RECENT_MYPYS = ["0.920", "0.910"]


async def validate_expected_success() -> None:
    """Check correctness of hardcoded Project.expected_success"""
    recent_mypy_exes = await asyncio.gather(
        *[
            setup_mypy(ARGS.base_dir / ("mypy_" + recent_mypy), recent_mypy)
            for recent_mypy in RECENT_MYPYS
        ]
    )

    async def inner(project: Project) -> Optional[str]:
        await project.setup()
        success = None
        for mypy_exe in recent_mypy_exes:
            mypy_result = await project.run_mypy(mypy_exe, typeshed_dir=None)
            if ARGS.debug:
                debug_print(format(Style.BLUE))
                debug_print(mypy_result)
                debug_print(format(Style.RESET))
            if mypy_result.success:
                success = mypy_exe
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
    """Check mypy's runtime over each project."""
    mypy_exe = await setup_mypy(ARGS.base_dir / "timer_mypy", RECENT_MYPYS[0])

    async def inner(project: Project) -> Tuple[float, Project]:
        await project.setup()
        start = time.time()
        await project.run_mypy(mypy_exe, typeshed_dir=None)
        end = time.time()
        return (end - start, project)

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
    assert not ARGS.new_typeshed
    assert not ARGS.old_typeshed

    mypy_exe = await setup_mypy(
        ARGS.base_dir / "bisect_mypy", revision_or_recent_tag_fn(ARGS.old), editable=True
    )
    repo_dir = ARGS.base_dir / "bisect_mypy" / "mypy"
    assert repo_dir.is_dir()

    projects = list(select_projects())
    await asyncio.wait([project.setup() for project in projects])

    async def run_wrapper(project: Project) -> Tuple[str, MypyResult]:
        return project.name, (await project.run_mypy(str(mypy_exe), typeshed_dir=None))

    results_fut = await asyncio.gather(*(run_wrapper(project) for project in projects))
    old_results: Dict[str, MypyResult] = dict(results_fut)
    if ARGS.debug:
        debug_print("\n".join(str(result) for result in old_results.values()))
        debug_print(format(Style.RESET))

    # Note git bisect start will clean up old bisection state
    await run(["git", "bisect", "start"], cwd=repo_dir, output=True)
    await run(["git", "bisect", "good"], cwd=repo_dir, output=True)
    new_revision = await get_revision_for_revision_or_date(ARGS.new or "origin/HEAD", repo_dir)
    await run(["git", "bisect", "bad", new_revision], cwd=repo_dir, output=True)

    def are_results_good(results: Dict[str, MypyResult]) -> bool:
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
        results: Dict[str, MypyResult] = dict(results_fut)

        state = "good" if are_results_good(results) else "bad"
        proc = await run(["git", "bisect", state], output=True, cwd=repo_dir)

        if "first bad commit" in proc.stdout:
            print(proc.stdout)
            return

        if ARGS.debug:
            debug_print("\n".join(str(result) for result in results.values()))
            debug_print(proc.stdout)
            debug_print(format(Style.RESET))


async def coverage() -> None:
    mypy_exe = await setup_mypy(ARGS.base_dir / "new_mypy", ARGS.new)

    projects = list(select_projects())
    mypy_python = mypy_exe.parent / "python"
    assert mypy_python.exists()

    all_paths = await asyncio.gather(
        *[project.source_paths(str(mypy_python)) for project in projects]
    )

    project_to_paths = {}
    project_to_lines = {}
    for project, paths in zip(projects, all_paths):
        project_to_paths[project.location] = len(paths)
        project_to_lines[project.location] = sum(map(line_count, paths))

    # for project in sorted(projects, key=lambda p: project_to_lines[p.location], reverse=True):
    #     p = project.location
    #     print(p, project_to_lines[p], project_to_paths[p])

    print(f"Checking {len(projects)} projects...")
    print(f"Containing {sum(project_to_paths.values())} files...")
    print(f"Totalling to {sum(project_to_lines.values())} lines...")


async def primer() -> int:
    projects = select_projects()
    new_mypy_revision = ARGS.new
    old_mypy_revision = revision_or_recent_tag_fn(ARGS.old)
    new_mypy, old_mypy = await setup_new_and_old_mypy(new_mypy_revision, old_mypy_revision)
    new_typeshed_dir, old_typeshed_dir = await setup_new_and_old_typeshed(
        ARGS.new_typeshed, ARGS.old_typeshed
    )

    results = [
        project.primer_result(
            new_mypy=str(new_mypy),
            new_mypy_revision=new_mypy_revision,
            old_mypy=str(old_mypy),
            old_mypy_revision=old_mypy_revision,
            new_typeshed=new_typeshed_dir,
            old_typeshed=old_typeshed_dir,
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


def parse_options(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    mypy_group = parser.add_argument_group("mypy")
    mypy_group.add_argument(
        "--new",
        help=(
            "new mypy version, defaults to HEAD "
            "(pypi version, anything commit-ish, or isoformatted date)"
        ),
    )
    mypy_group.add_argument(
        "--old",
        help=(
            "old mypy version, defaults to latest tag "
            "(pypi version, anything commit-ish, or isoformatted date)"
        ),
    )
    mypy_group.add_argument(
        "--repo",
        default="https://github.com/python/mypy",
        help=(
            "mypy repo to use (passed to git clone. if unspecified, we first try pypi, "
            "then fall back to github)"
        ),
    )
    mypy_group.add_argument(
        "--mypyc-compile-level",
        default=None,
        type=int,
        help="Compile mypy with the given mypyc optimisation level",
    )
    mypy_group.add_argument(
        "--custom-typeshed-repo",
        default="https://github.com/python/typeshed",
        help="typeshed repo to use (passed to git clone)",
    )
    mypy_group.add_argument(
        "--new-typeshed",
        help="new typeshed version, defaults to mypy's (anything commit-ish, or isoformatted date)",
    )
    mypy_group.add_argument(
        "--old-typeshed",
        help="old typeshed version, defaults to mypy's (anything commit-ish, or isoformatted date)",
    )

    proj_group = parser.add_argument_group("project selection")
    proj_group.add_argument(
        "-k", "--project-selector", help="regex to filter projects (matches against location)"
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
            "filter to hardcoded subset of projects where some recent mypy version succeeded "
            "aka are committed to the mypy way of life. also look at: --old-success"
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
        help="only output a result for a project if the old mypy run was successful",
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
        default=Path("/tmp/mypy_primer"),
        type=Path,
        help="dir to store repos and venvs",
    )
    primer_group.add_argument("--clear", action="store_true", help="delete repos and venvs")

    ret = parser.parse_args(argv)
    if (ret.num_shards is not None) != (ret.shard_index is not None):
        parser.error("--shard-index and --num-shards must be used together")
    return ret


ARGS: argparse.Namespace


def main() -> None:
    def inner() -> Optional[int]:
        global ARGS
        ARGS = parse_options(sys.argv[1:])

        if ARGS.base_dir.exists() and ARGS.clear:
            shutil.rmtree(ARGS.base_dir)
        ARGS.base_dir.mkdir(exist_ok=True)
        ARGS.projects_dir = ARGS.base_dir / "projects"
        ARGS.projects_dir.mkdir(exist_ok=True)

        coro: Awaitable[Optional[int]]
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


# ==============================
# project definitions
# ==============================


PROJECTS = [
    Project(
        location="https://github.com/python/mypy",
        mypy_cmd="{mypy} --config-file mypy_self_check.ini -p mypy -p mypyc",
        pip_cmd="{pip} install pytest types-typed-ast filelock",
        expected_success=True,
    ),
    Project(
        location="https://github.com/hauntsaninja/mypy_primer",
        mypy_cmd="{mypy} -m mypy_primer --strict",
        expected_success=True,
    ),
    Project(
        location="https://github.com/psf/black",
        mypy_cmd="{mypy} src",
        pip_cmd="{pip} install types-dataclasses types-typed-ast",
        mypyc_cmd=(
            "{python} setup.py --use-mypyc install "
            "&& {python} -m pip install pytest"
            "&& {python} -m pytest -k 'not incompatible_with_mypyc' tests/test_black.py"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/hauntsaninja/pyp",
        mypy_cmd="{mypy} --strict -m pyp",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pytest-dev/pytest",
        mypy_cmd="{mypy} src testing",
        pip_cmd="{pip} install py types-attrs types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pandas-dev/pandas",
        mypy_cmd="{mypy} pandas",
        pip_cmd=(
            "{pip} install numpy types-python-dateutil types-pytz types-PyMySQL "
            "types-setuptools pytest"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/pycqa/pylint",
        mypy_cmd="{mypy} pylint/checkers --ignore-missing-imports",
        pip_cmd="{pip} install types-toml",
        expected_success=True,
    ),
    Project(
        location="https://github.com/aio-libs/aiohttp",
        mypy_cmd="{mypy} aiohttp",
        pip_cmd="AIOHTTP_NO_EXTENSIONS=1 {pip} install -e . pytest",
        expected_success=True,
    ),
    Project(
        location="https://github.com/python-attrs/attrs",
        mypy_cmd=(
            "{mypy} src/attr/__init__.pyi src/attr/_version_info.pyi src/attr/converters.pyi"
            " src/attr/exceptions.pyi src/attr/filters.pyi src/attr/setters.pyi"
            " src/attr/validators.pyi tests/typing_example.py"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/sphinx-doc/sphinx",
        mypy_cmd="{mypy} sphinx",
        pip_cmd="{pip} install docutils-stubs types-typed-ast types-requests types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/scikit-learn/scikit-learn",
        mypy_cmd="{mypy} sklearn",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pypa/bandersnatch",
        mypy_cmd="{mypy} src",
        pip_cmd="{pip} install types-filelock types-freezegun types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/hauntsaninja/boostedblob",
        mypy_cmd="{mypy} boostedblob",
        expected_success=True,
    ),
    Project(
        location="https://github.com/quora/asynq",
        mypy_cmd="{mypy} asynq",
        expected_success=True,
    ),
    Project(
        location="https://github.com/scrapy/scrapy",
        mypy_cmd="{mypy} scrapy tests",
        pip_cmd="{pip} install types-attrs types-pyOpenSSL types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pypa/twine",
        mypy_cmd="{mypy} twine",
        pip_cmd="{pip} install keyring types-requests",
        expected_success=True,
    ),
    Project(
        location="https://github.com/more-itertools/more-itertools",
        mypy_cmd="{mypy} more_itertools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pydata/xarray",
        mypy_cmd="{mypy} .",
        pip_cmd="{pip} install types-PyYAML types-python-dateutil types-pytz",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pallets/werkzeug",
        mypy_cmd="{mypy} src/werkzeug tests",
        pip_cmd="{pip} install types-setuptools pytest",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pallets/jinja",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install markupsafe",
        expected_success=True,
    ),
    Project(
        location="https://github.com/bokeh/bokeh",
        mypy_cmd="{mypy} bokeh release",
        pip_cmd="{pip} install types-boto tornado",
        expected_success=True,
    ),
    Project(
        location="https://github.com/mystor/git-revise",
        mypy_cmd="{mypy} gitrevise",
        expected_success=True,
    ),
    Project(
        location="https://github.com/PyGithub/PyGithub",
        mypy_cmd="{mypy} github tests",
        pip_cmd="{pip} install types-requests",
        expected_success=True,
    ),
    Project(
        location="https://github.com/we-like-parsers/pegen",
        mypy_cmd="{mypy} src/pegen",
        expected_success=True,
    ),
    Project(
        location="https://github.com/zulip/zulip",
        mypy_cmd=(
            "{mypy} zerver zilencer zproject tools analytics corporate scripts --platform=linux"
        ),
        pip_cmd=(
            "{pip} install types-PyYAML types-polib types-redis types-Markdown types-decorator "
            "types-pytz types-requests types-python-dateutil types-orjson cryptography"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/dropbox/stone",
        mypy_cmd="{mypy} stone test",
        pip_cmd="{pip} install types-six",
        expected_success=True,
    ),
    Project(
        location="https://github.com/yelp/paasta",
        mypy_cmd="{mypy} paasta_tools",
        pip_cmd=(
            "{pip} install types-retry types-tzlocal types-ujson types-python-dateutil "
            "types-pytz types-PyYAML types-requests"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/PrefectHQ/prefect",
        mypy_cmd="{mypy} src",
        pip_cmd=(
            "{pip} install types-python-dateutil types-requests types-simplejson types-toml "
            "types-croniter types-PyYAML types-python-slugify types-pytz"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/pallets/itsdangerous",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install pytest",
        expected_success=True,
    ),
    Project(
        location="https://github.com/jab/bidict",
        mypy_cmd="{mypy} bidict",
        expected_success=True,
    ),
    Project(
        location="https://github.com/jaraco/zipp",
        mypy_cmd="{mypy} .",
        expected_success=True,
    ),
    Project(
        location="https://github.com/aaugustin/websockets",
        mypy_cmd="{mypy} --strict src",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pycqa/isort",
        mypy_cmd="{mypy} --ignore-missing-imports isort",
        pip_cmd="{pip} install types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/aio-libs/aioredis",
        mypy_cmd="{mypy} aioredis --ignore-missing-imports",
        expected_success=True,
    ),
    Project(
        location="https://github.com/agronholm/anyio",
        mypy_cmd="{mypy} src",
        expected_success=True,
    ),
    Project(
        location="https://github.com/aio-libs/yarl",
        mypy_cmd="{mypy} --show-error-codes yarl tests",
        pip_cmd="{pip} install multidict",
        expected_success=True,
    ),
    Project(
        location="https://github.com/freqtrade/freqtrade",
        mypy_cmd="{mypy} freqtrade scripts",
        pip_cmd=(
            "{pip} install types-cachetools types-requests types-python-dateutil types-tabulate "
            "types-filelock"
        ),
        expected_success=True,
    ),
    Project(
        location="https://github.com/google/jax",
        mypy_cmd="{mypy} jax",
        pip_cmd="{pip} install types-requests",
        expected_success=True,
    ),
    Project(
        location="https://github.com/dulwich/dulwich",
        mypy_cmd="{mypy} dulwich",
        pip_cmd="{pip} install types-certifi types-paramiko",
        expected_success=True,
    ),
    Project(
        location="https://github.com/optuna/optuna",
        mypy_cmd="{mypy} .",
        pip_cmd="{pip} install types-PyYAML types-redis types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/trailofbits/manticore",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install types-protobuf types-PyYAML types-redis types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/aiortc/aiortc",
        mypy_cmd="{mypy} src",
        expected_success=True,
    ),
    Project(
        location="https://github.com/willmcgugan/rich",
        mypy_cmd="{mypy} -p rich --ignore-missing-imports --warn-unreachable",
        pip_cmd="{pip} install types-attrs",
        expected_success=True,
    ),
    Project(
        location="https://github.com/dedupeio/dedupe",
        mypy_cmd="{mypy} --ignore-missing-imports dedupe",
        expected_success=True,
    ),
    Project(
        location="https://github.com/schemathesis/schemathesis",
        mypy_cmd="{mypy} src/schemathesis",
        pip_cmd="{pip} install types-attrs types-requests types-PyYAML",
        expected_success=True,
    ),
    Project(
        location="https://github.com/graphql-python/graphql-core",
        mypy_cmd="{mypy} src tests",
        expected_success=True,
    ),
    Project(
        location="https://github.com/Legrandin/pycryptodome",
        mypy_cmd="{mypy} lib",
        expected_success=True,
    ),
    Project(
        location="https://github.com/niklasf/python-chess",
        mypy_cmd="{mypy} --strict chess",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pytorch/ignite",
        mypy_cmd="{mypy}",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pypa/packaging",
        mypy_cmd="{mypy} packaging",
        expected_success=True,
    ),
    Project(
        location="https://github.com/samuelcolvin/pydantic",
        mypy_cmd="{mypy} pydantic",
        pip_cmd="{pip} install types-toml",
        expected_success=True,
    ),
    Project(
        location="https://github.com/encode/starlette",
        mypy_cmd="{mypy} starlette tests",
        pip_cmd="{pip} install types-requests types-PyYAML",
        expected_success=True,
    ),
    Project(
        location="https://github.com/aio-libs/janus",
        mypy_cmd="{mypy} janus --disallow-untyped-calls --disallow-incomplete-defs --strict",
        expected_success=True,
    ),
    Project(
        location="https://github.com/alerta/alerta",
        mypy_cmd="{mypy} alerta tests",
        pip_cmd="{pip} install types-PyYAML types-setuptools types-requests types-pytz",
        expected_success=True,
    ),
    Project(
        location="https://github.com/nolar/kopf",
        mypy_cmd="{mypy} kopf",
        pip_cmd="{pip} install types-setuptools types-PyYAML",
        expected_success=True,
    ),
    Project(
        location="https://github.com/davidhalter/parso",
        mypy_cmd="{mypy} parso",
        expected_success=True,
    ),
    Project(
        location="https://github.com/konradhalas/dacite",
        mypy_cmd="{mypy} dacite",
        expected_success=True,
    ),
    Project(
        location="https://github.com/ilevkivskyi/com2ann",
        mypy_cmd="{mypy} --python-version=3.8 src/com2ann.py src/test_com2ann.py",
        expected_success=True,
    ),
    Project(
        location="https://github.com/srittau/python-htmlgen",
        mypy_cmd="{mypy} htmlgen test_htmlgen",
        pip_cmd="{pip} install asserts",
        expected_success=True,
    ),
    Project(
        location="https://github.com/mitmproxy/mitmproxy",
        mypy_cmd="{mypy} .",
        expected_success=True,
    ),
    Project(
        location="https://github.com/jpadilla/pyjwt",
        mypy_cmd="{mypy} jwt",
        pip_cmd="{pip} install cryptography",
        expected_success=True,
    ),
    Project(
        location="https://github.com/apache/spark",
        mypy_cmd="{mypy} --config python/mypy.ini python/pyspark",
        pip_cmd="{pip} install numpy",
        expected_success=True,
    ),
    Project(
        location="https://github.com/laowantong/paroxython",
        mypy_cmd="{mypy} paroxython",
        pip_cmd="{pip} install types-typed-ast types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/Akuli/porcupine",
        mypy_cmd="{mypy} porcupine more_plugins",
        expected_success=True,
    ),
    # Project(
    #     location="https://github.com/edgedb/edgedb",
    #     mypy_cmd="{mypy} edb",
    #     # weeeee, extract the deps by noping out setuptools.setup and reading them
    #     # from the setup.py
    #     pip_cmd=(
    #         "{pip} install "
    #         '$(python3 -c "import setuptools; setuptools.setup=dict; '
    #         "from edb import buildmeta; buildmeta.get_version_from_scm = lambda *a: 1; "
    #         "import setup; "
    #         "print(' '.join(setup.TEST_DEPS+setup.DOCS_DEPS+setup.RUNTIME_DEPS))\")"
    #     ),
    #     expected_success=True,
    # ),
    Project(
        location="https://github.com/dropbox/mypy-protobuf",
        mypy_cmd="{mypy} mypy_protobuf/",
        pip_cmd="{pip} install types-protobuf",
        expected_success=True,
    ),
    # https://github.com/spack/spack/blob/develop/lib/spack/spack/cmd/style.py
    Project(
        location="https://github.com/spack/spack",
        mypy_cmd="{mypy} -p spack -p llnl",
        expected_success=True,
    ),
    Project(
        location="https://github.com/johtso/httpx-caching",
        mypy_cmd="{mypy} .",
        pip_cmd="{pip} install types-freezegun types-mock",
        expected_success=True,
    ),
    Project(
        location="https://github.com/python-poetry/poetry",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install types-requests",
        expected_success=True,
    ),
    Project(
        location="https://github.com/awslabs/sockeye",
        mypy_cmd=(
            "{mypy} --ignore-missing-imports --follow-imports=silent"
            " @typechecked-files --no-strict-optional"
        ),
        pip_cmd="{pip} install types-PyYAML",
        expected_success=True,
    ),
    Project(
        location="https://github.com/wntrblm/nox",
        mypy_cmd="{mypy} nox",
        pip_cmd="{pip} install jinja2 packaging importlib_metadata",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pandera-dev/pandera",
        mypy_cmd="{mypy} pandera tests",
        pip_cmd="{pip} install types-click types-PyYAML types-setuptools",
        expected_success=True,
    ),
    Project(
        location="https://github.com/pyodide/pyodide",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install packaging types-docutils types-pyyaml types-setuptools numpy",
        expected_success=True,
    ),
    Project(
        location="https://gitlab.com/cki-project/cki-lib",
        mypy_cmd="{mypy} --strict",
        pip_cmd="{pip} install types-PyYAML types-requests",
        expected_success=True,
    ),
    Project(
        location="https://github.com/python-jsonschema/check-jsonschema",
        mypy_cmd="{mypy} src",
        pip_cmd="{pip} install types-jsonschema types-requests",
        expected_success=True,
    ),
    *(
        [
            Project(
                location="https://github.com/sco1/pylox",
                mypy_cmd="{mypy} .",
                expected_success=True,
            ),
            Project(
                location="https://github.com/ppb/ppb-vector",
                mypy_cmd="{mypy} ppb_vector tests",
                pip_cmd="{pip} install hypothesis",
                expected_success=True,
            ),
        ]
        if sys.version_info >= (3, 10)
        else []
    ),
    # ==============================
    # Failures expected...
    # ==============================
    Project(
        location="https://github.com/pyppeteer/pyppeteer",
        mypy_cmd="{mypy} pyppeteer --config-file tox.ini",
        pip_cmd="{pip} install .",
    ),
    Project(
        location="https://github.com/pypa/pip",
        mypy_cmd="{mypy} src",
    ),
    Project(
        # relies on setup.py to create a version.py file
        location="https://github.com/pytorch/vision",
        mypy_cmd="{mypy}",
    ),
    # TODO: needs mypy-zope plugin
    # Project(
    #     location="https://github.com/twisted/twisted",
    #     mypy_cmd="{mypy} src",
    # ),
    # Other repos with plugins:
    # dry-python/returns, strawberry-graphql/strawberry, r-spacex/submanager, NeilGirdhar/efax
    Project(
        location="https://github.com/tornadoweb/tornado",
        mypy_cmd="{mypy} tornado",
        pip_cmd="{pip} install types-contextvars types-pycurl",
    ),
    Project(
        location="https://github.com/sympy/sympy",
        mypy_cmd="{mypy} sympy",
    ),
    Project(
        location="https://github.com/scipy/scipy",
        mypy_cmd="{mypy} scipy",
        pip_cmd="{pip} install numpy",
    ),
    Project(
        location="https://github.com/pycqa/flake8",
        mypy_cmd="{mypy} src tests",
        pip_cmd="{pip} install pytest",
    ),
    Project(
        location="https://github.com/home-assistant/core",
        mypy_cmd="{mypy} homeassistant",
        pip_cmd=(
            "{pip} install types-setuptools types-atomicwrites types-certifi types-croniter "
            "types-PyYAML types-requests types-attrs types-python-slugify types-backports"
        ),
    ),
    Project(
        location="https://github.com/kornia/kornia",
        mypy_cmd="{mypy} kornia",
    ),
    Project(
        location="https://github.com/ibis-project/ibis",
        mypy_cmd="{mypy} --ignore-missing-imports ibis",
        pip_cmd="{pip} install types-setuptools types-requests types-python-dateutil types-pytz",
    ),
    Project(
        location="https://github.com/streamlit/streamlit",
        mypy_cmd="{mypy} --config-file=lib/mypy.ini lib scripts",
        pip_cmd=(
            "{pip} install tornado packaging types-toml types-python-dateutil types-attrs "
            "types-setuptools types-protobuf types-pytz types-requests click pytest"
        ),
    ),
    Project(
        location="https://github.com/dragonchain/dragonchain",
        mypy_cmd="{mypy} dragonchain --error-summary",
        pip_cmd="{pip} install types-redis types-requests",
    ),
    Project(
        location="https://github.com/mikeshardmind/SinbadCogs",
        mypy_cmd="{mypy} .",
        pip_cmd="{pip} install types-pytz types-python-dateutil types-PyYAML types-attrs",
    ),
    Project(
        location="https://github.com/rotki/rotki",
        mypy_cmd="{mypy} rotkehlchen/ tools/data_faker",
        pip_cmd="{pip} install eth-typing types-requests types-setuptools",
    ),
    Project(
        location="https://github.com/arviz-devs/arviz",
        mypy_cmd="{mypy} .",
        pip_cmd="{pip} install pytest types-setuptools types-ujson numpy xarray",
    ),
    Project(
        location="https://github.com/urllib3/urllib3",
        mypy_cmd="{mypy} . --exclude setup.py",
        pip_cmd=(
            "{pip} install idna>=2.0.0 cryptography>=1.3.4 tornado>=6.1 pytest "
            "trustme==0.9.0 types-backports types-requests"
        ),
    ),
    Project(
        location="https://github.com/common-workflow-language/schema_salad",
        mypy_cmd="MYPYPATH=$MYPYPATH:typeshed {mypy} schema_salad",
        pip_cmd="{pip} install types-pkg_resources types-requests pytest ruamel.yaml",
    ),
    Project(
        location="https://github.com/common-workflow-language/cwltool",
        mypy_cmd="MYPYPATH=$MYPYPATH:typeshed {mypy} cwltool/*.py tests/*.py",
        pip_cmd="{pip} install types-requests types-setuptools types-psutil",
    ),
    Project(
        location="https://github.com/FasterSpeeding/Tanjun",
        mypy_cmd="{mypy} tanjun",
        pip_cmd="{pip} install hikari",
    ),
    Project(
        location="https://github.com/joerick/pyinstrument",
        mypy_cmd="{mypy} pyinstrument",
    ),
    Project(
        location="https://github.com/systemd/mkosi",
        mypy_cmd="{mypy} mkosi",
        pip_cmd="{pip} install cryptography",
    ),
    Project(
        location="https://github.com/Gobot1234/steam.py",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install cryptography",
    ),
    Project(
        location="https://github.com/cpitclaudel/alectryon",
        mypy_cmd="{mypy} alectryon.py",
    ),
    Project(
        location="https://github.com/yurijmikhalevich/rclip",
        mypy_cmd="{mypy} rclip",
    ),
    Project(
        location="https://github.com/psycopg/psycopg",
        mypy_cmd="{mypy}",
        pip_cmd="{pip} install pytest pproxy",
    ),
    Project(
        location="https://gitlab.com/dkg/python-sop",
        mypy_cmd="{mypy} --strict sop",
    ),
    Project(
        location="https://github.com/Rapptz/discord.py",
        mypy_cmd="{mypy} discord",
        pip_cmd="{pip} install types-requests types-setuptools aiohttp",
    ),
    Project(
        location="https://github.com/canonical/cloud-init",
        mypy_cmd="{mypy} cloudinit/ tests/ tools/",
        pip_cmd="{pip} install jinja2 pytest "
        "types-jsonschema types-oauthlib "
        "types-pyyaml types-requests types-setuptools",
    ),
]
assert len(PROJECTS) == len({p.name for p in PROJECTS})

if __name__ == "__main__":
    main()
