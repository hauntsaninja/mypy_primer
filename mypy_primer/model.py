from __future__ import annotations

import asyncio
import difflib
import os
import re
import shlex
import shutil
import string
import subprocess
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from mypy_primer.git_utils import ensure_repo_at_revision
from mypy_primer.globals import ctx
from mypy_primer.utils import Style, Venv, debug_print, has_uv, quote_path, run


@dataclass(frozen=True, kw_only=True)
class Project:
    location: str
    name_override: str | None = None

    mypy_cmd: str
    pyright_cmd: str | None

    pip_cmd: str | None = None
    deps: list[str] | None = None
    needs_mypy_plugins: bool = False

    # if expected_success, there is a recent version of mypy which passes cleanly
    expected_mypy_success: bool = False
    expected_pyright_success: bool = False

    # cost is vaguely proportional to type check time
    # for mypy we use the compiled times
    cost: dict[str, int] = field(default_factory=dict)

    revision: str | None = None
    min_python_version: tuple[int, int] | None = None
    supported_platforms: list[str] | None = None

    def __post_init__(self) -> None:
        if self.deps:
            assert all(d[0] in string.ascii_letters for d in self.deps)

    # custom __repr__ that omits defaults.
    def __repr__(self) -> str:
        result = f"Project(location={self.location!r}, mypy_cmd={self.mypy_cmd!r}"
        if self.name_override:
            result += f", name_override={self.name_override!r}"
        if self.pyright_cmd:
            result += f", pyright_cmd={self.pyright_cmd!r}"
        if self.pip_cmd:
            result += f", pip_cmd={self.pip_cmd!r}"
        if self.deps:
            result += f", deps={self.deps!r}"
        if self.needs_mypy_plugins:
            result += f", needs_mypy_plugins={self.needs_mypy_plugins!r}"
        if self.expected_mypy_success:
            result += f", expected_mypy_success={self.expected_mypy_success!r}"
        if self.expected_pyright_success:
            result += f", expected_pyright_success={self.expected_pyright_success!r}"
        if self.cost:
            result += f", cost={self.cost!r}"
        if self.revision:
            result += f", revision={self.revision!r}"
        if self.min_python_version:
            result += f", min_python_version={self.min_python_version!r}"
        if self.supported_platforms:
            result += f", supported_platforms={self.supported_platforms!r}"
        result += ")"
        return result

    @property
    def name(self) -> str:
        if self.name_override is not None:
            return self.name_override
        return Path(self.location).name

    @property
    def venv(self) -> Venv:
        return Venv(ctx.get().projects_dir / f"_{self.name}_venv")

    def expected_success(self, type_checker: str) -> bool:
        if type_checker == "mypy":
            return self.expected_mypy_success
        elif type_checker == "pyright":
            return self.expected_pyright_success
        else:
            raise ValueError(f"unknown type checker {type_checker}")

    def cost_for_type_checker(self, type_checker: str) -> int:
        default_cost = 5
        return self.cost.get(type_checker, default_cost)

    async def setup(self) -> None:
        if Path(self.location).exists():
            repo_dir = ctx.get().projects_dir / self.name
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
                self.location,
                ctx.get().projects_dir,
                self.revision,
                name_override=self.name_override,
            )
        assert repo_dir == ctx.get().projects_dir / self.name
        await self.venv.make_venv()
        if self.pip_cmd:
            assert "{pip}" in self.pip_cmd
            try:
                await run(
                    self.pip_cmd.format(pip=quote_path(self.venv.script("pip"))),
                    shell=True,
                    cwd=repo_dir,
                    output=True,
                )
            except subprocess.CalledProcessError as e:
                if e.output:
                    print(e.output)
                if e.stderr:
                    print(e.stderr)
                raise RuntimeError(f"pip install failed for {self.name}") from e
        if self.deps:
            if has_uv():
                install_base = f"uv pip install --python {quote_path(self.venv.python)}"
            else:
                install_base = f"{quote_path(self.venv.python)} -m pip install"
            install_cmd = f"{install_base} {' '.join(self.deps)}"
            try:
                await run(install_cmd, shell=True, cwd=repo_dir, output=True)
            except subprocess.CalledProcessError as e:
                if e.output:
                    print(e.output)
                if e.stderr:
                    print(e.stderr)
                raise RuntimeError(f"dependency install failed for {self.name}") from e

    def get_mypy_cmd(self, mypy: str | Path, additional_flags: Sequence[str] = ()) -> str:
        mypy_cmd = self.mypy_cmd
        assert "{mypy}" in self.mypy_cmd
        mypy_cmd = mypy_cmd.format(mypy=mypy)

        python_exe = self.venv.python
        mypy_cmd += f" --python-executable={quote_path(python_exe)}"
        if additional_flags:
            mypy_cmd += " " + " ".join(additional_flags)
        if ctx.get().output == "concise":
            mypy_cmd += "  --no-pretty --no-error-summary"

        mypy_cmd += " --warn-unused-ignores --warn-redundant-casts"
        mypy_cmd += (
            f" --no-incremental --cache-dir={os.devnull} --show-traceback --soft-error-limit=-1"
        )
        return mypy_cmd

    async def run_mypy(self, mypy: Path, typeshed_dir: Path | None) -> TypeCheckResult:
        additional_flags = ctx.get().additional_flags.copy()
        env = os.environ.copy()
        env["MYPY_FORCE_COLOR"] = "1"

        mypy_path = []  # TODO: this used to be exposed, could be useful to expose it again
        if typeshed_dir is not None:
            additional_flags.append(f"--custom-typeshed-dir={quote_path(typeshed_dir)}")
            mypy_path += list(map(str, typeshed_dir.glob("stubs/*")))

        if "MYPYPATH" in env:
            mypy_path = env["MYPYPATH"].split(os.pathsep) + mypy_path
        env["MYPYPATH"] = os.pathsep.join(mypy_path)
        if self.needs_mypy_plugins:
            env["MYPY_PRIMER_PLUGIN_SITE_PACKAGES"] = str(self.venv.site_packages)

        mypy_cmd = self.get_mypy_cmd(mypy, additional_flags)
        proc, runtime = await run(
            mypy_cmd,
            shell=True,
            output=True,
            check=False,
            cwd=ctx.get().projects_dir / self.name,
            env=env,
        )
        if ctx.get().debug:
            debug_print(f"{Style.BLUE}{mypy} on {self.name} took {runtime:.2f}s{Style.RESET}")

        output = proc.stderr + proc.stdout

        # Various logic to reduce noise in the diff
        if typeshed_dir is not None:
            # Differing line numbers and typeshed paths create noisy diffs.
            # Not a problem for stdlib because mypy silences errors from --custom-typeshed-dir.
            output = "".join(
                line
                for line in output.splitlines(keepends=True)
                if not line.startswith(str(typeshed_dir / "stubs"))
            )

        # Redact "note" lines which contain base_dir
        # Avoids noisy diffs when e.g., mypy points to a stub definition
        base_dir_re = (
            f"({re.escape(str(ctx.get().base_dir))}"
            f"|{re.escape(str(ctx.get().base_dir.resolve()))})"
            ".*: note:"
        )
        output = re.sub(base_dir_re, "note:", output)

        # Avoids some noise in tracebacks
        if "error: INTERNAL ERROR" in output:
            output = re.sub('File ".*/mypy', 'File "', output)

        return TypeCheckResult(
            mypy_cmd, output, not bool(proc.returncode), self.expected_mypy_success, runtime
        )

    def get_pyright_cmd(self, pyright: Path, additional_flags: Sequence[str] = ()) -> str:
        pyright_cmd = self.pyright_cmd or "{pyright}"
        assert "{pyright}" in pyright_cmd
        if additional_flags:
            pyright_cmd += " " + " ".join(additional_flags)
        pyright_cmd = pyright_cmd.format(pyright=pyright)
        return pyright_cmd

    async def run_pyright(self, pyright: Path, typeshed_dir: Path | None) -> TypeCheckResult:
        additional_flags: list[str] = []
        if typeshed_dir is not None:
            additional_flags.append(f"--typeshedpath {quote_path(typeshed_dir)}")
        pyright_cmd = self.get_pyright_cmd(pyright, additional_flags)
        pyright_cmd = f"{self.venv.activate}; {pyright_cmd}"
        proc, runtime = await run(
            pyright_cmd,
            shell=True,
            output=True,
            check=False,
            cwd=ctx.get().projects_dir / self.name,
        )
        if ctx.get().debug:
            debug_print(f"{Style.BLUE}{pyright} on {self.name} took {runtime:.2f}s{Style.RESET}")

        output = proc.stderr + proc.stdout
        return TypeCheckResult(
            pyright_cmd, output, not bool(proc.returncode), self.expected_pyright_success, runtime
        )

    async def run_typechecker(
        self, type_checker: Path, typeshed_dir: Path | None
    ) -> TypeCheckResult:
        if ctx.get().type_checker == "mypy":
            return await self.run_mypy(type_checker, typeshed_dir)
        elif ctx.get().type_checker == "pyright":
            return await self.run_pyright(type_checker, typeshed_dir)
        else:
            raise ValueError(f"Unknown type checker: {ctx.get().type_checker}")

    async def primer_result(
        self,
        new_type_checker: Path,
        old_type_checker: Path,
        new_typeshed: Path | None,
        old_typeshed: Path | None,
    ) -> PrimerResult:
        await self.setup()
        new_result, old_result = await asyncio.gather(
            self.run_typechecker(new_type_checker, new_typeshed),
            self.run_typechecker(old_type_checker, old_typeshed),
        )
        return PrimerResult(self, new_result, old_result)

    async def mypy_source_paths(self, mypy_python: str) -> list[Path]:
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
        proc, _ = await run(
            f"{mypy_python} -c {shlex.quote(program)} {mypy_cmd}",
            output=True,
            cwd=ctx.get().projects_dir / self.name,
            shell=True,
        )
        return [ctx.get().projects_dir / self.name / p for p in proc.stdout.splitlines()]

    @classmethod
    def from_location(cls, location: str) -> Project:
        additional_flags = ""
        if Path(location).is_file():
            with open(location, encoding="UTF-8") as f:
                header = f.readline().strip()
                if header.startswith("# flags:"):
                    additional_flags = header[len("# flags:") :]
        return Project(
            location=location, mypy_cmd=f"{{mypy}} {location} {additional_flags}", pyright_cmd=None
        )


@dataclass(frozen=True)
class TypeCheckResult:
    command: str
    output: str
    success: bool
    expected_success: bool
    runtime: float

    def __str__(self) -> str:
        ret = "> " + self.command + f" ({self.runtime:0.1f}s)\n"
        if self.expected_success and not self.success:
            ret += f"{Style.RED}{Style.BOLD}UNEXPECTED FAILURE{Style.RESET}\n"
        ret += textwrap.indent(self.output, "\t")
        return ret


@dataclass(frozen=True)
class PrimerResult:
    project: Project
    new_result: TypeCheckResult
    old_result: TypeCheckResult
    diff: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diff", self._get_diff())

    def _get_diff(self) -> str:
        d = difflib.Differ()

        old_output = self.old_result.output
        new_output = self.new_result.output

        old_lines = old_output.splitlines()
        new_lines = new_output.splitlines()

        # mypy's output appears to be nondeterministic for some same line errors, e.g. on pypa/pip
        # Work around that by ignoring identical removal and addition pairs, e.g.
        # "- a.py:1: error xyz" and "+ a.py:1: error xyz"
        diff_lines = [line for line in d.compare(old_lines, new_lines) if line[0] in ("+", "-")]
        net_change: dict[str, int] = defaultdict(int)
        for line in diff_lines:
            net_change[line[2:]] += 1 if line[0] == "+" else -1

        output_lines: list[str] = []
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
        return ret

    def format_concise(self) -> str:
        # runtime_diff = abs(self.new_result.runtime - self.old_result.runtime)
        runtime_ratio = self.new_result.runtime / self.old_result.runtime
        if runtime_ratio < 1:
            speed = "faster"
            runtime_ratio = 1 / runtime_ratio
        else:
            speed = "slower"

        has_runtime_diff = False  # runtime_diff > 10 and runtime_ratio > 1.05
        if not self.diff and not has_runtime_diff:
            return ""

        ret = f"{self.project.name} ({self.project.location})"
        if has_runtime_diff:
            ret += (
                f": typechecking got {runtime_ratio:.2f}x {speed} "
                f"({self.old_result.runtime:.1f}s -> {self.new_result.runtime:.1f}s)\n"
                "(Performance measurements are based on a single noisy sample)"
            )
        if self.diff:
            ret += "\n" + self.diff
        return ret

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
        ret += "old\n"
        ret += str(self.old_result)
        ret += "----------\n\n"
        ret += "new\n"
        ret += str(self.new_result)

        if self.diff:
            ret += "----------\n\n"
            ret += "diff\n"
            ret += textwrap.indent(self.diff, "\t")
            ret += "\n"

        ret += "==========\n"
        return ret
