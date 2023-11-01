from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydoctor_primer.globals import ctx

if sys.platform == "win32":
    import tempfile

    BIN_DIR = "scripts"
    PYDOCTOR_EXE_NAME = "pydoctor.exe"
    TEMP_DIR = tempfile.gettempdir()
else:
    BIN_DIR = "bin"
    PYDOCTOR_EXE_NAME = "pydoctor"
    TEMP_DIR = "/tmp"


if sys.platform == "win32":
    # shlex.quote() doesn't work on Windows

    ILLEGAL_PATH_CHARS = set('*?"<>')

    def quote_path(path: Path | str) -> str:
        path = str(path)
        if set(path) & ILLEGAL_PATH_CHARS:
            raise ValueError(
                'Illegal character in {path!r}: Windows paths cannot contain *, ?, ", <, or >'
            )
        return '"' + path + '"'

else:

    def quote_path(path: Path | str) -> str:
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
    return re.sub("\033\\[\\d+?m", "", text)


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
        start_t = time.perf_counter()
        stdout_b, stderr_b = await proc.communicate()
        end_t = time.perf_counter()

    stdout = stdout_b.decode("utf-8") if stdout_b is not None else None
    stderr = stderr_b.decode("utf-8") if stderr_b is not None else None
    assert proc.returncode is not None
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr), end_t - start_t


def line_count(path: Path) -> int:
    if path.is_dir():
        return 0
    buf_size = 1024 * 1024
    try:
        with open(path, "rb") as f:
            buf_iter = iter(lambda: f.raw.read(buf_size), b"")
            return sum(buf.count(b"\n") for buf in buf_iter)  # type: ignore
    except FileNotFoundError:
        return 0
