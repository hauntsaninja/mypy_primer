from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from mypy_primer.git_utils import RevisionLike, ensure_repo_at_revision
from mypy_primer.utils import BIN_DIR, MYPY_EXE_NAME, has_uv, make_venv, run


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

    await make_venv(venv_dir)

    async def pip_install(*targets: str) -> None:
        if has_uv():
            await run(
                ["uv", "pip", "install", "--python", str(venv_dir / BIN_DIR / "python"), *targets]
            )
        else:
            await run([str(venv_dir / BIN_DIR / "pip"), "install", *targets])

    if mypyc_compile_level is not None:
        editable = True

    install_from_repo = True
    if isinstance(revision_like, str) and not editable and repo is None:
        # optimistically attempt to install the revision of mypy we want from pypi
        try:
            await pip_install(f"mypy=={revision_like}")
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
            await pip_install("typing_extensions", "mypy_extensions")
            await run(
                [python_exe, "setup.py", "--use-mypyc", "build_ext", "--inplace"],
                cwd=repo_dir,
                env=env,
            )
        targets = []
        if editable:
            targets.append("--editable")
        targets.append(str(repo_dir))
        targets.append("tomli")
        await pip_install(*targets)

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

    pyright_exe = repo_dir / "packages" / "pyright" / "index.js"
    assert pyright_exe.exists()
    return pyright_exe


async def setup_typeshed(parent_dir: Path, *, repo: str, revision_like: RevisionLike) -> Path:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(exist_ok=True)
    return await ensure_repo_at_revision(repo, parent_dir, revision_like)
