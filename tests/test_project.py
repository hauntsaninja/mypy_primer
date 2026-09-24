import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from mypy_primer import utils
from mypy_primer.globals import ctx, parse_options
from mypy_primer.model import Project


@unittest.skipUnless(shutil.which("uv"), "uv is required")
class ProjectInstallTests(unittest.IsolatedAsyncioTestCase):
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

                    # Use a local wheel to keep installation offline and exercise UV_FIND_LINKS.
                    with ZipFile(root / "config_probe-1.0-py3-none-any.whl", "w") as wheel:
                        wheel.writestr(
                            "config_probe-1.0.dist-info/METADATA",
                            "Metadata-Version: 2.1\nName: config-probe\nVersion: 1.0\n",
                        )
                        wheel.writestr(
                            "config_probe-1.0.dist-info/WHEEL",
                            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                        )
                        wheel.writestr("config_probe-1.0.dist-info/RECORD", "")

                    args = parse_options([])
                    args.projects_dir = root / "projects"
                    args.projects_dir.mkdir()
                    token = ctx.set(args)
                    self.addCleanup(ctx.reset, token)
                    project = Project(
                        location=str(source),
                        mypy_cmd=None,
                        pyright_cmd=None,
                        install_cmd="{install} config-probe" if custom_install else None,
                        deps=None if custom_install else ["config-probe"],
                    )
                    # Seed packages are unnecessary here and would require registry access.
                    subprocess.run(
                        [
                            "uv",
                            "venv",
                            "--no-config",
                            "--offline",
                            "--python",
                            sys.executable,
                            str(project.venv.dir),
                        ],
                        check=True,
                    )
                    env = {
                        "UV_NO_CONFIG": "0",
                        "UV_NO_INDEX": "1",
                        "UV_OFFLINE": "1",
                        "UV_FIND_LINKS": str(root),
                    }
                    with patch.dict(os.environ, env), patch.object(utils.Venv, "make_venv"):
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
