from __future__ import annotations

import asyncio
import difflib
import os
import re
import shlex
import shutil
import subprocess
import textwrap
import venv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from mypy_primer.git_utils import ensure_repo_at_revision
from mypy_primer.globals import ctx
from mypy_primer.utils import BIN_DIR, Style, debug_print, run


@dataclass(frozen=True)
class Project:
    location: str
    mypy_cmd: str
    revision: str | None = None
    pip_cmd: str | None = None
    # if expected_success, there is a recent version of mypy which passes cleanly
    expected_success: bool = False
    # cost is vaguely proportional to type check time
    cost: int = 3

    @property
    def name(self) -> str:
        return Path(self.location).name

    @property
    def venv_dir(self) -> Path:
        return ctx.get().projects_dir / f"_{self.name}_venv"

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
                self.location, ctx.get().projects_dir, self.revision
            )
        assert repo_dir == ctx.get().projects_dir / self.name
        if self.pip_cmd:
            assert "{pip}" in self.pip_cmd
            venv.create(self.venv_dir, with_pip=True, clear=True)
            try:
                await run(
                    self.pip_cmd.format(pip=str(self.venv_dir / BIN_DIR / "pip")),
                    shell=True,
                    cwd=repo_dir,
                    output=True,
                )
            except subprocess.CalledProcessError as e:
                if e.output:
                    print(e.output)
                if e.stderr:
                    print(e.stderr)
                raise RuntimeError(f"pip install failed for {self.location}") from e

    def get_mypy_cmd(self, mypy: str | Path, additional_flags: Sequence[str] = ()) -> str:
        mypy_cmd = self.mypy_cmd
        assert "{mypy}" in self.mypy_cmd
        if self.pip_cmd:
            python_exe = self.venv_dir / BIN_DIR / "python"
            mypy_cmd += f" --python-executable={python_exe}"
        if additional_flags:
            mypy_cmd += " " + " ".join(additional_flags)
        if ctx.get().output == "concise":
            mypy_cmd += "  --no-pretty --no-error-summary"
        mypy_cmd += (
            f" --no-incremental --cache-dir={os.devnull} --show-traceback --soft-error-limit=-1"
        )
        mypy_cmd = mypy_cmd.format(mypy=mypy)
        return mypy_cmd

    async def run_mypy(
        self, mypy: str | Path, typeshed_dir: Path | None, mypy_path: list[str]
    ) -> MypyResult:
        additional_flags = ctx.get().additional_flags.copy()
        env = os.environ.copy()
        env["MYPY_FORCE_COLOR"] = "1"

        mypy_path = mypy_path.copy()
        if typeshed_dir is not None:
            additional_flags.append(f"--custom-typeshed-dir={typeshed_dir}")
            mypy_path += list(map(str, typeshed_dir.glob("stubs/*")))

        if "MYPYPATH" in env:
            mypy_path = env["MYPYPATH"].split(os.pathsep) + mypy_path
        env["MYPYPATH"] = os.pathsep.join(mypy_path)

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
        if typeshed_dir is not None:
            # Differing line numbers and typeshed paths create noisy diffs.
            # Not a problem for stdlib because mypy silences errors from --custom-typeshed-dir.
            output = "".join(
                line
                for line in output.splitlines(keepends=True)
                if not line.startswith(str(typeshed_dir / "stubs"))
            )
        return MypyResult(
            mypy_cmd, output, not bool(proc.returncode), self.expected_success, runtime
        )

    async def primer_result(
        self,
        new_mypy: str,
        old_mypy: str,
        new_typeshed: Path | None,
        old_typeshed: Path | None,
        new_mypypath: list[str],
        old_mypypath: list[str],
    ) -> PrimerResult:
        await self.setup()
        new_result, old_result = await asyncio.gather(
            self.run_mypy(new_mypy, new_typeshed, new_mypypath),
            self.run_mypy(old_mypy, old_typeshed, old_mypypath),
        )
        return PrimerResult(self, new_result, old_result)

    async def source_paths(self, mypy_python: str) -> list[Path]:
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
        return Project(location=location, mypy_cmd=f"{{mypy}} {location} {additional_flags}")


@dataclass(frozen=True)
class MypyResult:
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
    new_result: MypyResult
    old_result: MypyResult
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
        # Hide "note" lines which contain ctx.get().base_dir... this hides differences between
        # file paths, e.g., when mypy points to a stub definition.
        base_dir_re = re.compile(f"{re.escape(str(ctx.get().base_dir))}.*: note:")
        old_lines = [line for line in old_lines if not re.search(base_dir_re, line)]
        new_lines = [line for line in new_lines if not re.search(base_dir_re, line)]
        diff = d.compare(old_lines, new_lines)
        diff_lines = [line for line in diff if line[0] in ("+", "-")]

        # mypy's output appears to be nondeterministic for some same line errors, e.g. on pypa/pip
        # Work around that by ignoring identical removal and addition pairs, e.g.
        # "- a.py:1: error xyz" and "+ a.py:1: error xyz"
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
        runtime_diff = abs(self.new_result.runtime - self.old_result.runtime)
        runtime_ratio = self.new_result.runtime / self.old_result.runtime
        if runtime_ratio < 1:
            speed = "faster"
            runtime_ratio = 1 / runtime_ratio
        else:
            speed = "slower"

        has_runtime_diff = runtime_diff > 5 and runtime_ratio > 1.05
        if not self.diff and not has_runtime_diff:
            return ""

        ret = f"{self.project.name} ({self.project.location})"
        if has_runtime_diff:
            ret += (
                f" got {runtime_ratio:.2f}x {speed} "
                f"({self.old_result.runtime:.1f}s -> {self.new_result.runtime:.1f}s)"
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
