from __future__ import annotations

import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable

from mypy_primer.utils import run

RevisionLike = str | None | Callable[[Path], Awaitable[str]]


# repo_source could be a URL *or* a local path
async def clone(repo_source: str, *, repo_dir: Path, cwd: Path, shallow: bool = False) -> None:
    if os.path.exists(repo_source):
        repo_source = os.path.abspath(repo_source)
    cmd = ["git", "clone", "--recurse-submodules", repo_source, str(repo_dir)]
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
    proc, _ = await run(
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


async def ensure_repo_at_revision(
    repo_url: str, cwd: Path, revision_like: RevisionLike, *, name_override: str | None = None
) -> Path:
    if name_override:
        repo_dir = cwd / name_override
    else:
        repo_dir = cwd / Path(repo_url).name
    if repo_dir.is_dir():
        await refresh(repo_dir)
    else:
        await clone(repo_url, repo_dir=repo_dir, cwd=cwd, shallow=revision_like is None)
    assert repo_dir.is_dir(), f"{repo_dir} is not a directory"

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
    proc, _ = await run(["git", "rev-list", "--tags", "-1"], output=True, cwd=repo_dir)
    proc, _ = await run(
        ["git", "describe", "--tags", proc.stdout.strip()], output=True, cwd=repo_dir
    )
    return proc.stdout.strip()


def revision_or_recent_tag_fn(revision: str | None) -> RevisionLike:
    return revision if revision is not None else get_recent_tag
