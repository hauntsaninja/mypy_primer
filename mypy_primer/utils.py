from __future__ import annotations

import asyncio
import functools
import re
import shlex
import shutil
import subprocess
import sys
import time
import venv
from enum import Enum
from pathlib import Path
from typing import Any

from mypy_primer.globals import ctx

if sys.platform == "win32":
    import tempfile

    TEMP_DIR = tempfile.gettempdir()
else:
    TEMP_DIR = "/tmp"


if sys.platform == "win32":
    # shlex.quote() doesn't work on Windows

    ILLEGAL_PATH_CHARS = set('*?"<>')

    def quote_path(path: Path) -> str:
        path = str(path)
        if set(path) & ILLEGAL_PATH_CHARS:
            raise ValueError(
                'Illegal character in {path!r}: Windows paths cannot contain *, ?, ", <, or >'
            )
        return '"' + path + '"'

else:

    def quote_path(path: Path) -> str:
        return shlex.quote(str(path))


class Style(str, Enum):
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # In Python 3.11, Enum.__format__ by default returns "Style.RED",
    # while previously it returned the value of the enum member. Once
    # we support only 3.11, we can inherit from enum.StrEnum and remove
    # this method.
    def __format__(self, format_spec: str) -> str:
        return self.value


def strip_colour_code(text: str) -> str:
    return re.sub("\x1b(\\[\\d*?m|\\(.)", "", text)


def debug_print(obj: Any) -> None:
    assert ctx.get().debug
    print(obj, file=sys.stderr)


_semaphore: asyncio.Semaphore | None = None


async def run(
    cmd: str | list[str],
    *,
    shell: bool = False,
    output: bool = False,
    check: bool = True,
    **kwargs: Any,
) -> tuple[subprocess.CompletedProcess[str], float]:
    if output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    else:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)

    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.BoundedSemaphore(ctx.get().concurrency)
    async with _semaphore:
        if ctx.get().debug:
            log = cmd if shell else shlex.join(cmd)
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
        start_t = time.perf_counter()
        stdout_b, stderr_b = await proc.communicate()
        end_t = time.perf_counter()

    stdout = stdout_b.decode("utf-8") if stdout_b is not None else None
    stderr = stderr_b.decode("utf-8") if stderr_b is not None else None
    assert proc.returncode is not None
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr), end_t - start_t


@functools.cache
def has_uv() -> bool:
    return bool(shutil.which("uv"))


class Venv:
    def __init__(self, dir: Path) -> None:
        self.dir = dir

    @property
    def bin(self) -> Path:
        if sys.platform == "win32":
            BIN_DIR = "scripts"
        else:
            BIN_DIR = "bin"
        return self.dir / BIN_DIR

    def script(self, name: str) -> Path:
        if sys.platform == "win32":
            return self.bin / f"{name}.exe"
        else:
            return self.bin / name

    @property
    def python(self) -> Path:
        return self.script("python")

    @property
    def site_packages(self) -> Path:
        if sys.platform == "win32":
            return self.dir / "Lib" / "site-packages"
        else:
            pyname = f"python{sys.version_info.major}.{sys.version_info.minor}"
            return self.dir / "lib" / pyname / "site-packages"

    @property
    def activate_cmd(self) -> str:
        if sys.platform == "win32":
            return str(self.bin / "activate.bat")
        else:
            return f". {self.bin / 'activate'}"

    async def make_venv(self) -> None:
        if has_uv():
            await run(["uv", "venv", str(self.dir), "--python", sys.executable, "--seed"])
        else:
            venv.create(self.dir, with_pip=True, clear=True)


def line_count(path: Path) -> int:
    if path.is_dir():
        return 0
    buf_size = 1024 * 1024
    try:
        with open(path, "rb") as f:
            buf_iter = iter(lambda: f.raw.read(buf_size), b"")
            return sum(buf.count(b"\n") for buf in buf_iter)
    except FileNotFoundError:
        return 0


@functools.cache
def get_npm() -> str:
    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("'npm' is not found.")
    if sys.platform == "win32":
        # On Windows, npm is typically installed as 'npm.cmd'
        # and `subprocess` requires full name for it to run.
        return Path(npm_path).name
    else:
        return "npm"
