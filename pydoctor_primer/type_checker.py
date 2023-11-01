from __future__ import annotations

import subprocess
import venv
from pathlib import Path

from pydoctor_primer.git_utils import RevisionLike, ensure_repo_at_revision
from pydoctor_primer.utils import BIN_DIR, PYDOCTOR_EXE_NAME, run


async def setup_pydoctor(
    pydoctor_dir: Path,
    revision_like: RevisionLike,
    *,
    repo: str | None,
    editable: bool = False,
) -> Path:
    pydoctor_dir.mkdir(exist_ok=True)
    venv_dir = pydoctor_dir / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    pip_exe = str(venv_dir / BIN_DIR / "pip")

    install_from_repo = True
    if isinstance(revision_like, str) and not editable and repo is None:
        # optimistically attempt to install the revision of pydoctor we want from pypi
        try:
            await run([pip_exe, "install", f"pydoctor=={revision_like}"])
            install_from_repo = False
        except subprocess.CalledProcessError:
            install_from_repo = True
    if install_from_repo:
        if repo is None:
            repo = "https://github.com/twisted/pydoctor"
        repo_dir = await ensure_repo_at_revision(repo, pydoctor_dir, revision_like)

        install_cmd = [pip_exe, "install"]
        if editable:
            install_cmd.append("--editable")
        install_cmd.append(str(repo_dir))
        await run(install_cmd)

    pydoctor_exe = venv_dir / BIN_DIR / PYDOCTOR_EXE_NAME
    assert pydoctor_exe.exists()
    return pydoctor_exe
