from __future__ import annotations

import enum
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mypy_primer.git_utils import RevisionLike, ensure_repo_at_revision
from mypy_primer.utils import Venv, get_npm, has_uv, run


async def setup_mypy(
    mypy_dir: Path,
    *,
    revision_like: RevisionLike,
    repo: str | None,
    mypyc_compile_level: int | None,
    editable: bool = False,
) -> Path:
    mypy_dir.mkdir(exist_ok=True)
    venv = Venv(mypy_dir / "venv")
    await venv.make_venv()

    async def pip_install(*targets: str) -> None:
        if has_uv():
            await run(["uv", "pip", "install", "--python", str(venv.python), *targets])
        else:
            await run([str(venv.python), "-m", "pip", "install", *targets])

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
            env["MYPY_USE_MYPYC"] = "1"
            env["MYPYC_OPT_LEVEL"] = str(mypyc_compile_level)  # can be zero
            await pip_install(
                "typing_extensions",
                "mypy_extensions",
                "tomli",
                "pathspec",
                "setuptools",
                "types-psutil",
                "types-setuptools",
            )
            try:
                await run(
                    [str(venv.python), "-m", "pip", "install", ".", "--no-build-isolation"],
                    cwd=repo_dir,
                    env=env,
                    output=True,
                )
            except subprocess.CalledProcessError as e:
                print("Error while building mypy", file=sys.stderr)
                print(e.stdout, file=sys.stderr)
                print(e.stderr, file=sys.stderr)
                raise e
        else:
            targets = []
            if editable:
                targets.append("--editable")
            targets.append(str(repo_dir))
            targets.append("tomli")
            targets.append("pathspec")
            await pip_install(*targets)

    with open(venv.site_packages / "primer_plugin.pth", "w") as f:
        # pth file that lets us let mypy import plugins from another venv
        # importantly, this puts the plugin paths at the back of sys.path, so they cannot
        # clobber mypy or its dependencies
        f.write(
            r"""import os; import sys; exec('''env = os.environ.get("MYPY_PRIMER_PLUGIN_SITE_PACKAGES")\nif env: sys.path.extend(env.split(os.pathsep))''')"""
        )

    mypy_exe = venv.script("mypy")
    if sys.platform == "darwin":
        # warm up mypy on macos to avoid the first run being slow
        await run([str(mypy_exe), "--version"])
    assert mypy_exe.exists()
    return mypy_exe


async def setup_pyright(
    pyright_dir: Path, *, revision_like: RevisionLike, repo: str | None
) -> Path:
    pyright_dir.mkdir(exist_ok=True)

    if repo is None:
        repo = "https://github.com/microsoft/pyright"
    repo_dir = await ensure_repo_at_revision(repo, pyright_dir, revision_like)

    npm = get_npm()
    await run([npm, "run", "install:all"], cwd=repo_dir)
    await run([npm, "run", "build"], cwd=repo_dir / "packages" / "pyright")

    pyright_exe = repo_dir / "packages" / "pyright" / "index.js"
    assert pyright_exe.exists()
    return pyright_exe


async def setup_ty(
    ty_dir: Path,
    revision_like: RevisionLike,
    *,
    build_mode: RustBuildMode,
    repo: str | None,
) -> Path:
    ty_dir.mkdir(parents=True, exist_ok=True)

    if repo is None:
        repo = "https://github.com/astral-sh/ruff"
    repo_dir = await ensure_repo_at_revision(repo, ty_dir, revision_like)

    cargo_target_dir = ty_dir / "target"
    if not os.environ.get("MYPY_PRIMER_NO_REBUILD", False):
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(cargo_target_dir)

        try:
            await run(
                ["cargo", "build", "--bin", "ty", *build_mode.flags()],
                cwd=repo_dir,
                env=env,
                output=True,
            )
        except subprocess.CalledProcessError as e:
            print("Error while building 'ty'", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            raise e

    ty_exe = cargo_target_dir / build_mode.artifact_directory() / "ty"
    assert ty_exe.exists()
    return ty_exe


async def setup_pyrefly(
    pyrefly_dir: Path,
    revision_like: RevisionLike,
    *,
    build_mode: RustBuildMode,
    repo: str | None,
    typeshed_dir: Path | None,
) -> Path:
    pyrefly_dir.mkdir(parents=True, exist_ok=True)

    if repo is None:
        repo = "https://github.com/facebook/pyrefly"
    repo_dir = await ensure_repo_at_revision(repo, pyrefly_dir, revision_like)

    env = os.environ.copy()
    if typeshed_dir is not None:
        if typeshed_dir.name != "typeshed":
            raise RuntimeError(f"Unexpected typeshed dir {typeshed_dir}")
        env["TYPESHED_ROOT"] = str(typeshed_dir.parent)

    if not os.environ.get("MYPY_PRIMER_NO_REBUILD", False):
        try:
            await run(
                ["cargo", "build", *build_mode.flags()],
                cwd=repo_dir / "pyrefly",
                env=env,
                output=True,
            )
        except subprocess.CalledProcessError as e:
            print("Error while building 'pyrefly'", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            raise e

    pyrefly_exe = repo_dir / "target" / build_mode.artifact_directory() / "pyrefly"
    assert pyrefly_exe.exists()
    return pyrefly_exe


async def setup_typeshed(parent_dir: Path, *, repo: str, revision_like: RevisionLike) -> Path:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(exist_ok=True)
    return await ensure_repo_at_revision(repo, parent_dir, revision_like)


class RustBuildMode(enum.Enum):
    RELEASE = "release"
    DEBUG = "debug"

    def flags(self) -> list[str]:
        if self == RustBuildMode.RELEASE:
            return ["--release"]
        else:
            return []

    def artifact_directory(self) -> Path:
        if self == RustBuildMode.RELEASE:
            return Path("release")
        else:
            return Path("debug")
