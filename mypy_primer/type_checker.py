from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from mypy_primer.git_utils import RevisionLike, ensure_repo_at_revision
from mypy_primer.utils import BIN_DIR, MYPY_EXE_NAME, run


async def setup_mypy(
    mypy_dir: Path,
    revision_like: RevisionLike,
    *,
    repo: str | None,
    mypyc_compile_level: int | None,
    editable: bool = False,
) -> Path:
    mypy_dir.mkdir(exist_ok=True)
    venv_dir = mypy_dir / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    pip_exe = str(venv_dir / BIN_DIR / "pip")

    if mypyc_compile_level is not None:
        editable = True

    install_from_repo = True
    if isinstance(revision_like, str) and not editable and repo is None:
        # optimistically attempt to install the revision of mypy we want from pypi
        try:
            await run([pip_exe, "install", f"mypy=={revision_like}"])
            install_from_repo = False
        except subprocess.CalledProcessError:
            install_from_repo = True

    if install_from_repo:
        if repo is None:
            repo = "https://github.com/python/mypy"
        repo_dir = await ensure_repo_at_revision(repo, mypy_dir, revision_like)
        if mypyc_compile_level is not None:
            env = os.environ.copy()
            env["MYPYC_OPT_LEVEL"] = str(mypyc_compile_level)
            python_exe = str(venv_dir / BIN_DIR / "python")
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

    mypy_exe = venv_dir / BIN_DIR / MYPY_EXE_NAME
    if sys.platform == "darwin":
        # warm up mypy on macos to avoid the first run being slow
        await run([str(mypy_exe), "--version"])
    assert mypy_exe.exists()
    return mypy_exe


async def setup_pyright(
    pyright_dir: Path,
    revision_like: RevisionLike,
    *,
    repo: str | None,
) -> Path:
    pyright_dir.mkdir(exist_ok=True)

    if repo is None:
        repo = "https://github.com/microsoft/pyright"
    repo_dir = await ensure_repo_at_revision(repo, pyright_dir, revision_like)

    await run(["npm", "run", "install:all"], cwd=repo_dir)
    await run(["npm", "run", "build"], cwd=repo_dir / "packages" / "pyright")
    return repo_dir / "packages" / "pyright" / "index.js"


async def setup_typeshed(parent_dir: Path, *, repo: str, revision_like: RevisionLike) -> Path:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(exist_ok=True)
    return await ensure_repo_at_revision(repo, parent_dir, revision_like)
