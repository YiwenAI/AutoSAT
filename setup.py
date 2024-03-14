import os
from setuptools import find_packages, setup

setup(
    name="autosat",
    packages=[package for package in find_packages() if package.startswith("autosat")],
    install_requires=[
        "ray[all] == 2.8.0",
        "jinja2 == 3.1.2",
        "openai == 0.28.0",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            "mypy",
            # Lint code (flake8 replacement)
            "ruff",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx>=5.3,<7.0",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
    },
    description="implementations of automatically optimizing SAT Solvers via LLMs.",
    author="Yiwen Sun, Xianyin Zhang, Shiyu Huang",
    url="https://github.com/YiwenAI/AutoSAT",
    author_email="ywsun22@m.fudan.edu.cn",
    keywords="LLMs SAT solvers automatic",
    license="MIT",
    version="0.1",
    python_requires=">=3.8",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/YiwenAI/AutoSAT",
        "Paper": "https://arxiv.org/abs/2402.10705",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)