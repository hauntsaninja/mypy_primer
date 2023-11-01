from setuptools import find_packages, setup

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()

setup(
    name="pydoctor_primer",
    version="0.1.0",
    author="Shantanu Jain",
    author_email="hauntsaninja@gmail.com",
    description="Run pydoctor on many open source project for the purpose of evaluating changes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tristanlatr/pydoctor_primer",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
    ],
    packages=find_packages(),
    entry_points={"console_scripts": ["pydoctor_primer=pydoctor_primer.main:main"]},
    python_requires=">=3.7",
)
