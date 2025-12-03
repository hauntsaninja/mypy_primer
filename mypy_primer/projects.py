import pathlib
import tomllib

from mypy_primer.model import Project

PROJECTS_PATH = pathlib.Path(__file__).parent / "projects.toml"


def get_projects() -> list[Project]:
    with PROJECTS_PATH.open("rb") as f:
        data = tomllib.load(f)

    projects: list[Project] = []
    for entry in data["projects"]:
        if "expected_success" in entry:
            entry["expected_success"] = tuple(entry["expected_success"])
        if "min_python_version" in entry:
            py_maj, py_min = entry["min_python_version"].split(".")
            entry["min_python_version"] = int(py_maj), int(py_min)

        _ = entry.setdefault("mypy_cmd", None)
        _ = entry.setdefault("pyright_cmd", None)

        projects.append(Project(**entry))

    assert projects, "no projects found"
    assert len(projects) == len({p.name for p in projects})
    for p in projects:
        assert p.supported_platforms is None or all(
            p in ("linux", "darwin") for p in p.supported_platforms
        )

    return projects
