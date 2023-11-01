from __future__ import annotations

import difflib
import subprocess
import sys

from pydoctor_primer.model import Project

def update_projects(projects: list[Project], check: bool = False) -> None:
    # modifies `get_projects` in place.
    result = []
    with open(__file__) as f:
        keep = True
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]
            if line == "    projects = [":
                result.append(f"    projects = {projects!r}")
                keep = False
            if keep:
                result.append(line)
            if line == "    ]":
                keep = True

    if check:
        code_proc = subprocess.run(
            ["black", "-"], input="\n".join(result), capture_output=True, text=True
        )
        code_proc.check_returncode()
        code = code_proc.stdout

        with open(__file__) as f:
            in_file = f.read()
            if in_file != code:
                diff = difflib.context_diff(
                    in_file.splitlines(keepends=True),
                    code.splitlines(keepends=True),
                    fromfile=__file__,
                    tofile=__file__,
                )
                print("".join(diff))
                sys.exit(1)
    else:
        with open(__file__, "w") as f:
            f.write("\n".join(result))


def get_projects() -> list[Project]:
    projects = [
        Project(
            location="https://github.com/twisted/pydoctor",
            pydoctor_cmd="{pydoctor} ./pydoctor --privacy='HIDDEN:pydoctor.test'", 
            expected_success=True
        ),
        Project(
            location="https://github.com/twisted/twisted",
            pydoctor_cmd="{pydoctor} ./src/twisted --docformat=plaintext", 
            expected_success=True
        ),
    ]
    assert len(projects) == len({p.name for p in projects})
    return projects
