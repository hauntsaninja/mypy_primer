from __future__ import annotations

import asyncio
import difflib
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import venv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from pydoctor_primer.git_utils import ensure_repo_at_revision
from pydoctor_primer.globals import ctx
from pydoctor_primer.utils import BIN_DIR, Style, debug_print, quote_path, run

extra_dataclass_args = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **extra_dataclass_args)
class Project:
    location: str
    pydoctor_cmd: str
    revision: str | None = None
    min_python_version: tuple[int, int] | None = None
    pip_cmd: str | None = None
    # if expected_success, there is a recent version of pydoctor which passes cleanly
    expected_success: bool = False
    name_override: str | None = None

    # custom __repr__ that omits defaults.
    def __repr__(self) -> str:
        result = f"Project(location={self.location!r}, pydoctor_cmd={self.pydoctor_cmd!r}"
        if self.pip_cmd:
            result += f", pip_cmd={self.pip_cmd!r}"
        if self.expected_success:
            result += f", expected_success={self.expected_success!r}"
        if self.revision:
            result += f", revision={self.revision!r}"
        if self.min_python_version:
            result += f", min_python_version={self.min_python_version!r}"
        if self.name_override:
            result += f", name_override={self.name_override!r}"
        result += ")"
        return result

    @property
    def name(self) -> str:
        if self.name_override is not None:
            return self.name_override
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
                self.location,
                ctx.get().projects_dir,
                self.revision,
                name_override=self.name_override,
            )
        assert repo_dir == ctx.get().projects_dir / self.name
        if self.pip_cmd:
            assert "{pip}" in self.pip_cmd
            venv.create(self.venv_dir, with_pip=True, clear=True)
            try:
                await run(
                    self.pip_cmd.format(pip=quote_path(self.venv_dir / BIN_DIR / "pip")),
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

    def get_pydoctor_cmd(self, pydoctor_path: str | Path, additional_flags: Sequence[str] = ()) -> str:
        cmd = self.pydoctor_cmd
        assert "{pydoctor}" in self.pydoctor_cmd
        cmd = cmd.format(pydoctor=pydoctor_path)

        if additional_flags:
            cmd += " " + " ".join(additional_flags)

        cmd += " --make-html --quiet"
        return cmd

    async def run_pydoctor(self, pydoctor_path: str | Path) -> TypeCheckResult:
        additional_flags = ctx.get().additional_flags.copy()
        env = os.environ.copy()

        cmd = self.get_pydoctor_cmd(pydoctor_path, additional_flags)
        proc, runtime = await run(
            cmd,
            shell=True,
            output=True,
            check=False,
            cwd=ctx.get().projects_dir / self.name,
            env=env,
        )
        if ctx.get().debug:
            debug_print(f"{Style.BLUE}{pydoctor_path} on {self.name} took {runtime:.2f}s{Style.RESET}")

        output = proc.stderr + proc.stdout

        # Redact lines which contain base_dir
        # Avoids noisy diffs
        base_dir_re = (
            f"({re.escape(str(ctx.get().base_dir))}"
            f"|{re.escape(str(ctx.get().base_dir.resolve()))})"
        )
        output = re.sub(base_dir_re, "", output)

        return TypeCheckResult(
            cmd, output, not bool(proc.returncode), self.expected_success, runtime
        )

    async def run_typechecker(
        self, type_checker: str | Path, 
    ) -> TypeCheckResult:
        return await self.run_pydoctor(type_checker)

    async def primer_result(
        self,
        new_type_checker: str,
        old_type_checker: str,
    ) -> PrimerResult:
        await self.setup()
        new_result, old_result = await asyncio.gather(
            self.run_typechecker(new_type_checker),
            self.run_typechecker(old_type_checker),
        )
        return PrimerResult(self, new_result, old_result)

    @classmethod
    def from_location(cls, location: str) -> Project:
        return Project(location=location, pydoctor_cmd=f"{{pydoctor}} {location}")


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
        runtime_diff = abs(self.new_result.runtime - self.old_result.runtime)
        runtime_ratio = self.new_result.runtime / self.old_result.runtime
        if runtime_ratio < 1:
            speed = "faster"
            runtime_ratio = 1 / runtime_ratio
        else:
            speed = "slower"

        has_runtime_diff = runtime_diff > 10 and runtime_ratio > 1.05
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
