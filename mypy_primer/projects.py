from __future__ import annotations

import sys

from mypy_primer.model import Project


def get_projects() -> list[Project]:
    projects = [
        Project(
            location="https://github.com/python/mypy",
            mypy_cmd="{mypy} --config-file mypy_self_check.ini -p mypy -p mypyc",
            pip_cmd="{pip} install pytest types-typed-ast filelock",
            expected_mypy_success=True,
            mypy_cost=20,
        ),
        Project(
            location="https://github.com/hauntsaninja/mypy_primer",
            mypy_cmd="{mypy} -p mypy_primer --strict",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/psf/black",
            mypy_cmd="{mypy} src",
            pip_cmd="{pip} install types-dataclasses types-typed-ast aiohttp click "
            "tomli platformdirs",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/hauntsaninja/pyp",
            mypy_cmd="{mypy} --strict -m pyp",
            expected_mypy_success=True,
            pyright_cmd="{pyright}",
        ),
        Project(
            location="https://github.com/pytest-dev/pytest",
            mypy_cmd="{mypy} src testing",
            pip_cmd="{pip} install attrs py types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pandas-dev/pandas",
            mypy_cmd="{mypy} pandas",
            pip_cmd=(
                "{pip} install numpy types-python-dateutil types-pytz types-PyMySQL "
                "types-setuptools pytest"
            ),
            expected_mypy_success=True,
            mypy_cost=120,
        ),
        Project(
            location="https://github.com/pycqa/pylint",
            mypy_cmd="{mypy} pylint/checkers --ignore-missing-imports",
            pip_cmd="{pip} install types-toml",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/aio-libs/aiohttp",
            mypy_cmd="{mypy} aiohttp",
            pip_cmd="AIOHTTP_NO_EXTENSIONS=1 {pip} install -e . pytest",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/python-attrs/attrs",
            mypy_cmd=(
                "{mypy} src/attr/__init__.pyi src/attr/_version_info.pyi src/attr/converters.pyi"
                " src/attr/exceptions.pyi src/attr/filters.pyi src/attr/setters.pyi"
                " src/attr/validators.pyi tests/typing_example.py"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/sphinx-doc/sphinx",
            mypy_cmd="{mypy} sphinx",
            pip_cmd="{pip} install docutils-stubs types-typed-ast types-requests types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/scikit-learn/scikit-learn",
            mypy_cmd="{mypy} sklearn",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pypa/bandersnatch",
            mypy_cmd="{mypy} src",
            pip_cmd="{pip} install types-filelock types-freezegun types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/hauntsaninja/boostedblob",
            mypy_cmd="{mypy} boostedblob",
            pip_cmd="{pip} install aiohttp uvloop pycryptodome",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/quora/asynq",
            mypy_cmd="{mypy} asynq",
            pip_cmd="{pip} install qcore",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/scrapy/scrapy",
            mypy_cmd="{mypy} scrapy tests",
            pip_cmd="{pip} install attrs types-pyOpenSSL types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pypa/twine",
            mypy_cmd="{mypy} twine",
            pip_cmd="{pip} install keyring types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/more-itertools/more-itertools",
            mypy_cmd="{mypy} more_itertools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pydata/xarray",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install types-PyYAML types-python-dateutil types-pytz",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pallets/werkzeug",
            mypy_cmd="{mypy} src/werkzeug tests",
            pip_cmd="{pip} install types-setuptools pytest markupsafe",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pallets/jinja",
            mypy_cmd="{mypy}",
            pip_cmd="{pip} install markupsafe",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/mystor/git-revise",
            mypy_cmd="{mypy} gitrevise",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/PyGithub/PyGithub",
            mypy_cmd="{mypy} github tests",
            pip_cmd="{pip} install types-requests pyjwt",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/we-like-parsers/pegen",
            mypy_cmd="{mypy} src/pegen",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/zulip/zulip",
            mypy_cmd=(
                "{mypy} zerver zilencer zproject tools analytics corporate scripts --platform=linux"
            ),
            pip_cmd=(
                "{pip} install types-PyYAML types-polib types-redis types-Markdown types-decorator "
                "types-pytz types-requests types-python-dateutil types-orjson cryptography"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/dropbox/stone",
            mypy_cmd="{mypy} stone test",
            pip_cmd="{pip} install types-six",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/yelp/paasta",
            mypy_cmd="{mypy} paasta_tools",
            pip_cmd=(
                "{pip} install types-retry types-tzlocal types-ujson types-python-dateutil "
                "types-pytz types-PyYAML types-requests"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/PrefectHQ/prefect",
            mypy_cmd="{mypy} src",
            pip_cmd=(
                "{pip} install types-python-dateutil types-requests types-simplejson types-toml "
                "types-croniter types-PyYAML types-python-slugify types-pytz cryptography"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pallets/itsdangerous",
            mypy_cmd="{mypy}",
            pip_cmd="{pip} install pytest",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/jab/bidict",
            mypy_cmd="{mypy} bidict",
            expected_mypy_success=True,
            pyright_cmd="{pyright}",
        ),
        Project(
            location="https://github.com/jaraco/zipp",
            mypy_cmd="{mypy} .",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/aaugustin/websockets",
            mypy_cmd="{mypy} --strict src",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pycqa/isort",
            mypy_cmd="{mypy} --ignore-missing-imports isort",
            pip_cmd="{pip} install types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/aio-libs/aioredis",
            mypy_cmd="{mypy} aioredis --ignore-missing-imports",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/agronholm/anyio",
            mypy_cmd="{mypy} src",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/aio-libs/yarl",
            mypy_cmd="{mypy} --show-error-codes yarl tests",
            pip_cmd="{pip} install multidict",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/freqtrade/freqtrade",
            mypy_cmd="{mypy} freqtrade scripts",
            pip_cmd=(
                "{pip} install types-cachetools types-requests types-python-dateutil "
                "types-tabulate types-filelock"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/google/jax",
            mypy_cmd="{mypy} jax",
            pip_cmd="{pip} install types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/dulwich/dulwich",
            mypy_cmd="{mypy} dulwich",
            pip_cmd="{pip} install types-certifi types-paramiko",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/optuna/optuna",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install types-PyYAML types-redis types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/trailofbits/manticore",
            mypy_cmd="{mypy}",
            pip_cmd="{pip} install types-protobuf types-PyYAML types-redis types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/aiortc/aiortc",
            mypy_cmd="{mypy} src",
            pip_cmd="{pip} install cryptography",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/Textualize/rich",
            mypy_cmd="{mypy} -p rich --ignore-missing-imports --warn-unreachable",
            pip_cmd="{pip} install attrs",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/dedupeio/dedupe",
            mypy_cmd="{mypy} --ignore-missing-imports dedupe",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/schemathesis/schemathesis",
            mypy_cmd="{mypy} src/schemathesis",
            pip_cmd="{pip} install attrs types-requests types-PyYAML",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/graphql-python/graphql-core",
            mypy_cmd="{mypy} src tests",
            expected_mypy_success=True,
            mypy_cost=70,
        ),
        Project(
            location="https://github.com/Legrandin/pycryptodome",
            mypy_cmd="{mypy} lib",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/niklasf/python-chess",
            mypy_cmd="{mypy} --strict chess",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pytorch/ignite",
            mypy_cmd="{mypy}",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pypa/packaging",
            mypy_cmd="{mypy} packaging",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/samuelcolvin/pydantic",
            mypy_cmd="{mypy} pydantic",
            pip_cmd="{pip} install types-toml",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/encode/starlette",
            mypy_cmd="{mypy} starlette tests",
            pip_cmd="{pip} install types-requests types-PyYAML",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/aio-libs/janus",
            mypy_cmd="{mypy} janus --disallow-untyped-calls --disallow-incomplete-defs --strict",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/alerta/alerta",
            mypy_cmd="{mypy} alerta tests",
            pip_cmd="{pip} install types-PyYAML types-setuptools types-requests types-pytz",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/nolar/kopf",
            mypy_cmd="{mypy} kopf",
            pip_cmd="{pip} install types-setuptools types-PyYAML",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/davidhalter/parso",
            mypy_cmd="{mypy} parso",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/konradhalas/dacite",
            mypy_cmd="{mypy} dacite",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/ilevkivskyi/com2ann",
            mypy_cmd="{mypy} --python-version=3.8 src/com2ann.py src/test_com2ann.py",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/srittau/python-htmlgen",
            mypy_cmd="{mypy} htmlgen test_htmlgen",
            pip_cmd="{pip} install asserts",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/mitmproxy/mitmproxy",
            mypy_cmd="{mypy} .",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/jpadilla/pyjwt",
            mypy_cmd="{mypy} jwt",
            pip_cmd="{pip} install cryptography",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/apache/spark",
            mypy_cmd="{mypy} --config python/mypy.ini python/pyspark",
            pip_cmd="{pip} install numpy",
            expected_mypy_success=True,
            mypy_cost=20,
        ),
        Project(
            location="https://github.com/laowantong/paroxython",
            mypy_cmd="{mypy} paroxython",
            pip_cmd="{pip} install types-typed-ast types-setuptools",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/Akuli/porcupine",
            mypy_cmd="{mypy} porcupine more_plugins",
            expected_mypy_success=True,
        ),
        # Project(
        #     location="https://github.com/edgedb/edgedb",
        #     mypy_cmd="{mypy} edb",
        #     # weeeee, extract the deps by noping out setuptools.setup and reading them
        #     # from the setup.py
        #     pip_cmd=(
        #         "{pip} install "
        #         '$(python3 -c "import setuptools; setuptools.setup=dict; '
        #         "from edb import buildmeta; buildmeta.get_version_from_scm = lambda *a: 1; "
        #         "import setup; "
        #         "print(' '.join(setup.TEST_DEPS+setup.DOCS_DEPS+setup.RUNTIME_DEPS))\")"
        #     ),
        #     expected_mypy_success=True,
        # ),
        Project(
            location="https://github.com/dropbox/mypy-protobuf",
            mypy_cmd="{mypy} mypy_protobuf/",
            pip_cmd="{pip} install types-protobuf",
            expected_mypy_success=True,
        ),
        # https://github.com/spack/spack/blob/develop/lib/spack/spack/cmd/style.py
        Project(
            location="https://github.com/spack/spack",
            mypy_cmd="{mypy} -p spack -p llnl",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/johtso/httpx-caching",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install types-freezegun types-mock",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/python-poetry/poetry",
            mypy_cmd="{mypy}",
            pip_cmd="{pip} install types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/awslabs/sockeye",
            mypy_cmd=(
                "{mypy} --ignore-missing-imports --follow-imports=silent"
                " @typechecked-files --no-strict-optional"
            ),
            pip_cmd="{pip} install types-PyYAML",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/wntrblm/nox",
            mypy_cmd="{mypy} nox",
            pip_cmd="{pip} install jinja2 packaging importlib_metadata",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pandera-dev/pandera",
            mypy_cmd="{mypy} pandera tests",
            pip_cmd="{pip} install types-click types-PyYAML types-setuptools types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://gitlab.com/cki-project/cki-lib",
            mypy_cmd="{mypy} --strict",
            pip_cmd="{pip} install types-PyYAML types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/python-jsonschema/check-jsonschema",
            mypy_cmd="{mypy} src",
            pip_cmd="{pip} install types-jsonschema types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/pybind/pybind11",
            mypy_cmd="{mypy} --exclude '^(tests|docs)/' .",
            pip_cmd="{pip} install nox rich",
            expected_mypy_success=True,
        ),
        Project(
            "https://github.com/rpdelaney/downforeveryone",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install types-requests types-requests",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/DataDog/dd-trace-py",
            mypy_cmd="{mypy}",
            pip_cmd=(
                "{pip} install attrs types-six types-setuptools types-docutils "
                "types-PyYAML types-protobuf"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/systemd/mkosi",
            mypy_cmd="{mypy} mkosi",
            pip_cmd="{pip} install cryptography",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/sympy/sympy",
            mypy_cmd="{mypy} sympy",
            expected_mypy_success=True,
            mypy_cost=70,
        ),
        Project(
            location="https://github.com/nion-software/nionutils",
            mypy_cmd="{mypy} --namespace-packages --strict -p nion.utils",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/PyCQA/flake8-pyi",
            mypy_cmd="{mypy} pyi.py",
            pip_cmd="{pip} install types-pyflakes",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/internetarchive/openlibrary",
            mypy_cmd="{mypy} openlibrary",
            pip_cmd=(
                "{pip} install types-PyYAML types-python-dateutil types-requests "
                "types-simplejson types-Deprecated"
            ),
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/JohannesBuchner/imagehash",
            mypy_cmd="{mypy} imagehash",
            pip_cmd="{pip} install numpy types-Pillow",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/Kalmat/PyWinCtl",
            mypy_cmd="{mypy} src/pywinctl",
            pip_cmd="{pip} install types-setuptools types-pywin32 types-python-xlib",
            expected_mypy_success=True,
        ),
        Project(
            location="https://github.com/mesonbuild/meson",
            mypy_cmd="./run_mypy.py --mypy {mypy}",
            pip_cmd="{pip} install types-PyYAML",
            expected_mypy_success=True,
        ),
        *(
            [
                Project(
                    location="https://github.com/sco1/pylox",
                    mypy_cmd="{mypy} .",
                    pip_cmd="{pip} install attrs",
                    expected_mypy_success=True,
                ),
                Project(
                    location="https://github.com/ppb/ppb-vector",
                    mypy_cmd="{mypy} ppb_vector tests",
                    pip_cmd="{pip} install hypothesis",
                    expected_mypy_success=True,
                ),
            ]
            if sys.version_info >= (3, 10)
            else []
        ),
        # ==============================
        # Failures expected...
        # ==============================
        Project(
            location="https://github.com/pyppeteer/pyppeteer",
            mypy_cmd="{mypy} pyppeteer --config-file tox.ini",
            pip_cmd="{pip} install .",
        ),
        Project(
            location="https://github.com/pypa/pip",
            mypy_cmd="{mypy} src",
        ),
        Project(
            # relies on setup.py to create a version.py file
            location="https://github.com/pytorch/vision",
            mypy_cmd="{mypy}",
        ),
        # TODO: needs mypy-zope plugin
        # Project(
        #     location="https://github.com/twisted/twisted",
        #     mypy_cmd="{mypy} src",
        # ),
        # Other repos with plugins:
        # dry-python/returns, strawberry-graphql/strawberry, r-spacex/submanager, NeilGirdhar/efax
        Project(
            location="https://github.com/tornadoweb/tornado",
            mypy_cmd="{mypy} tornado",
            pip_cmd="{pip} install types-contextvars types-pycurl",
        ),
        Project(
            location="https://github.com/scipy/scipy",
            mypy_cmd="{mypy} scipy",
            pip_cmd="{pip} install numpy",
        ),
        Project(
            location="https://github.com/pycqa/flake8",
            mypy_cmd="{mypy} src tests",
            pip_cmd="{pip} install pytest",
        ),
        Project(
            location="https://github.com/home-assistant/core",
            mypy_cmd="{mypy} homeassistant",
            pip_cmd=(
                "{pip} install attrs pydantic "
                "types-setuptools types-atomicwrites types-certifi types-croniter "
                "types-PyYAML types-requests types-python-slugify types-backports"
            ),
            mypy_cost=70,
        ),
        Project(
            location="https://github.com/kornia/kornia",
            mypy_cmd="{mypy} kornia",
        ),
        Project(
            location="https://github.com/ibis-project/ibis",
            mypy_cmd="{mypy} --ignore-missing-imports ibis",
            pip_cmd="{pip} install types-setuptools types-requests "
            "types-python-dateutil types-pytz",
        ),
        Project(
            location="https://github.com/streamlit/streamlit",
            mypy_cmd="{mypy} --config-file=lib/mypy.ini lib scripts",
            pip_cmd=(
                "{pip} install attrs tornado packaging types-toml types-python-dateutil "
                "types-setuptools types-protobuf types-pytz types-requests types-cffi click pytest"
            ),
        ),
        Project(
            location="https://github.com/dragonchain/dragonchain",
            mypy_cmd="{mypy} dragonchain --error-summary",
            pip_cmd="{pip} install types-redis types-requests",
        ),
        Project(
            location="https://github.com/mikeshardmind/SinbadCogs",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install attrs types-pytz types-python-dateutil types-PyYAML",
        ),
        Project(
            location="https://github.com/rotki/rotki",
            mypy_cmd="{mypy} rotkehlchen/ tools/data_faker",
            pip_cmd="{pip} install eth-typing types-requests types-setuptools",
        ),
        Project(
            location="https://github.com/arviz-devs/arviz",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install pytest types-setuptools types-ujson numpy xarray",
            mypy_cost=20,
        ),
        Project(
            location="https://github.com/urllib3/urllib3",
            mypy_cmd="{mypy} . --exclude setup.py",
            pip_cmd=(
                "{pip} install idna>=2.0.0 cryptography>=1.3.4 tornado>=6.1 pytest "
                "trustme==0.9.0 types-backports types-requests"
            ),
        ),
        Project(
            location="https://github.com/common-workflow-language/schema_salad",
            mypy_cmd="MYPYPATH=$MYPYPATH:mypy-stubs {mypy} schema_salad",
            pip_cmd="{pip} install types-pkg_resources types-requests "
            "types-dataclasses types-setuptools black pytest ruamel.yaml",
        ),
        Project(
            location="https://github.com/common-workflow-language/cwltool",
            mypy_cmd="MYPYPATH=$MYPYPATH:mypy-stubs {mypy} cwltool/*.py tests/*.py",
            pip_cmd="{pip} install types-requests types-setuptools types-psutil "
            "types-mock cwl-utils schema-salad ruamel-yaml pytest pytest-httpserver",
            mypy_cost=20,
        ),
        Project(
            location="https://github.com/FasterSpeeding/Tanjun",
            mypy_cmd="{mypy} tanjun",
            pip_cmd="{pip} install hikari alluka",
        ),
        Project(
            location="https://github.com/joerick/pyinstrument",
            mypy_cmd="{mypy} pyinstrument",
        ),
        Project(
            location="https://github.com/Gobot1234/steam.py",
            mypy_cmd="{mypy} steam",
            pip_cmd="{pip} install cryptography",
        ),
        Project(
            location="https://github.com/cpitclaudel/alectryon",
            mypy_cmd="{mypy} alectryon.py",
        ),
        Project(
            location="https://github.com/yurijmikhalevich/rclip",
            mypy_cmd="{mypy} rclip",
        ),
        Project(
            location="https://github.com/psycopg/psycopg",
            mypy_cmd="{mypy}",
            pip_cmd="{pip} install pytest pproxy",
        ),
        Project(
            location="https://gitlab.com/dkg/python-sop",
            mypy_cmd="{mypy} --strict sop",
        ),
        Project(
            location="https://github.com/Rapptz/discord.py",
            mypy_cmd="{mypy} discord",
            pip_cmd="{pip} install types-requests types-setuptools aiohttp",
            mypy_cost=20,
        ),
        Project(
            location="https://github.com/canonical/cloud-init",
            mypy_cmd="{mypy} cloudinit/ tests/ tools/",
            pip_cmd=(
                "{pip} install jinja2 pytest "
                "types-jsonschema types-oauthlib "
                "types-pyyaml types-requests types-setuptools"
            ),
            mypy_cost=20,
        ),
        Project(
            location="https://github.com/mongodb/mongo-python-driver",
            mypy_cmd="{mypy} bson gridfs tools pymongo",
            pip_cmd="{pip} install types-requests types-pyOpenSSL cryptography certifi",
        ),
        Project(
            location="https://github.com/artigraph/artigraph",
            mypy_cmd="{mypy}",
            pip_cmd="{pip} install pydantic numpy pytest",
        ),
        Project(
            location="https://github.com/MaterializeInc/materialize",
            mypy_cmd="MYPYPATH=$MYPYPATH:misc/python {mypy} ci misc/python",
            pip_cmd="{pip} install -r ci/builder/requirements.txt",
            mypy_cost=30,
        ),
        Project(
            "https://github.com/canonical/operator",
            mypy_cmd="{mypy} ops",
            pip_cmd="{pip} install types-PyYAML",
        ),
        Project(
            "https://github.com/astropenguin/xarray-dataclasses",
            mypy_cmd="{mypy} xarray_dataclasses",
            pip_cmd="{pip} install numpy xarray",
        ),
        Project(
            "https://github.com/caronc/apprise",
            mypy_cmd="{mypy} .",
            pip_cmd=(
                "{pip} install types-six types-mock cryptography types-requests "
                "types-PyYAML types-Markdown pytest certifi"
            ),
        ),
        Project(
            "https://github.com/daveleroy/sublime_debugger",
            mypy_cmd="{mypy} modules --namespace-packages",
            pip_cmd="{pip} install certifi",
        ),
        Project(
            "https://github.com/Finistere/antidote",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install pytest",
        ),
        Project(
            "https://github.com/cognitedata/Expression",
            mypy_cmd="{mypy} .",
            pip_cmd="{pip} install pytest",
        ),
        Project(
            location="https://github.com/pyodide/pyodide",
            mypy_cmd="{mypy} src pyodide-build --exclude 'setup.py|^src/tests|conftest.py'",
            pip_cmd="{pip} install packaging types-docutils types-pyyaml types-setuptools numpy",
        ),
        Project(
            location="https://github.com/bokeh/bokeh",
            mypy_cmd="{mypy} src release",
            pip_cmd="{pip} install types-boto tornado numpy jinja2 selenium",
        ),
        Project(
            location="https://github.com/pandas-dev/pandas-stubs",
            mypy_cmd="{mypy} pandas-stubs tests",
            mypy_cost=20,
            pyright_cmd="{pyright}",
            expected_pyright_success=True,
        ),
        Project(
            location="https://github.com/enthought/comtypes",
            mypy_cmd="{mypy} comtypes --platform win32",
            pip_cmd="{pip} install numpy",
        ),
        Project(
            location="https://github.com/mit-ll-responsible-ai/hydra-zen",
            mypy_cmd="{mypy} src",
            pip_cmd="{pip} install pydantic beartype hydra-core",
            mypy_cost=30,
        ),
        Project(
            location="https://github.com/Avasam/Auto-Split",
            mypy_cmd="{mypy} src",
            pip_cmd=(
                "{pip} install certifi ImageHash numpy packaging PyQt6 "
                "types-d3dshot types-keyboard types-Pillow types-psutil types-PyAutoGUI "
                "types-pyinstaller types-pywin32 types-requests types-toml"
            ),
        ),
        Project(
            location="https://github.com/Avasam/speedrun.com_global_scoreboard_webapp",
            mypy_cmd="{mypy} backend",
            pip_cmd=(
                "{pip} install Flask PyJWT requests-cache types-Flask-SQLAlchemy "
                "types-httplib2 types-requests"
            ),
            mypy_cost=30,
        ),
        Project(
            location="https://github.com/pwndbg/pwndbg",
            mypy_cmd="{mypy} pwndbg",
            pip_cmd="{pip} install types-gdb",
        ),
        Project(
            location="https://github.com/keithasaurus/koda-validate",
            mypy_cmd="{mypy} koda_validate --strict",
            pip_cmd="{pip} install koda",
        ),
    ]
    assert len(projects) == len({p.name for p in projects})
    return projects
