import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from mypy_primer import model, type_checker
from mypy_primer.globals import ctx, parse_options


def _write_wheel(root: Path, name: str) -> Path:
    wheel_path = root / f"{name}-1.0-py3-none-any.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(
            f"{name}-1.0.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
        )
        wheel.writestr(
            f"{name}-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr(f"{name}-1.0.dist-info/RECORD", "")
    return wheel_path


@unittest.skipUnless(shutil.which("uv"), "uv is required")
class InstallTests(unittest.IsolatedAsyncioTestCase):
    async def test_install_ignores_uv_config(self) -> None:
        for config_name in ("pyproject.toml", "uv.toml"):
            for custom_install in (False, True):
                with (
                    self.subTest(config_name=config_name, custom_install=custom_install),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = root / "source"
                    source.mkdir()
                    config = 'required-version = "==0.0.0"\n'
                    if config_name == "pyproject.toml":
                        config = "[tool.uv]\n" + config
                    (source / config_name).write_text(config)

                    user_config = root / "user-config" / "uv" / "uv.toml"
                    user_config.parent.mkdir(parents=True)
                    user_config.write_text('required-version = "==0.0.0"\n')

                    # Local wheels keep venv seeding and dependency installation offline.
                    for name in ("pip", "setuptools", "wheel", "config_probe"):
                        _write_wheel(root, name)

                    args = parse_options([])
                    args.projects_dir = root / "projects"
                    args.projects_dir.mkdir()
                    token = ctx.set(args)
                    self.addCleanup(ctx.reset, token)
                    project = model.Project(
                        location=str(source),
                        mypy_cmd=None,
                        pyright_cmd=None,
                        install_cmd="{install} config-probe" if custom_install else None,
                        deps=None if custom_install else ["config-probe"],
                    )
                    env = {
                        "UV_NO_CONFIG": "0",
                        "UV_NO_INDEX": "1",
                        "UV_OFFLINE": "1",
                        "UV_FIND_LINKS": str(root),
                        "XDG_CONFIG_HOME": str(user_config.parent.parent),
                        "APPDATA": str(user_config.parent.parent),
                    }
                    with patch.dict(os.environ, env):
                        await project.setup()
                        self.assertEqual(os.environ["UV_NO_CONFIG"], "0")
                    self.assertTrue(
                        (
                            project.venv.site_packages / "config_probe-1.0.dist-info/METADATA"
                        ).is_file()
                    )
                    self.assertEqual(
                        (args.projects_dir / project.name / config_name).read_text(), config
                    )

    async def test_mypy_install_ignores_uv_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user_config = root / "user-config" / "uv" / "uv.toml"
            user_config.parent.mkdir(parents=True)
            user_config.write_text('required-version = "==0.0.0"\n')
            for name in ("pip", "setuptools", "wheel"):
                _write_wheel(root, name)

            # A minimal stand-in exercises installation and the macOS warmup offline.
            with ZipFile(_write_wheel(root, "mypy"), "a") as wheel:
                wheel.writestr("mypy.py", 'def main():\n    print("mypy 1.0")\n')
                wheel.writestr(
                    "mypy-1.0.dist-info/entry_points.txt", "[console_scripts]\nmypy = mypy:main\n"
                )

            token = ctx.set(parse_options([]))
            self.addCleanup(ctx.reset, token)
            env = {
                "UV_NO_CONFIG": "0",
                "UV_NO_INDEX": "1",
                "UV_OFFLINE": "1",
                "UV_FIND_LINKS": str(root),
                "XDG_CONFIG_HOME": str(user_config.parent.parent),
                "APPDATA": str(user_config.parent.parent),
            }
            with (
                patch.dict(os.environ, env),
                patch.object(
                    type_checker,
                    "ensure_repo_at_revision",
                    side_effect=AssertionError("the local mypy wheel should be installable"),
                ),
            ):
                executable = await type_checker.setup_mypy(
                    root / "mypy", revision_like="1.0", repo=None, mypyc_compile_level=None
                )
                self.assertEqual(os.environ["UV_NO_CONFIG"], "0")
            self.assertTrue(executable.is_file())
