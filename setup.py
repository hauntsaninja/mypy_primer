from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mypy_primer",
    version="0.1.0",
    author="Shantanu Jain",
    author_email="hauntsaninja@gmail.com",
    description="Run mypy over millions of lines of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hauntsaninja/mypy_primer",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
    ],
    py_modules=["mypy_primer"],
    entry_points={"console_scripts": ["mypy_primer=mypy_primer:main"]},
    python_requires=">=3.7",
)
