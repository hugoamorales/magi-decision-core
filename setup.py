"""
File: setup.py
Description: Setup script for MAGI Decision Core v5 package.
Modifications:
- Added internal documentation comments.
"""
"""
Setup script for installing the MAGI Decision Core v5 package.

This script defines the package metadata, dependencies, and entry points for the MAGI system.
Run `pip install .` to install the package locally, or `python setup.py sdist bdist_wheel` to create a distributable package.
"""
from setuptools import setup

setup(
    name="magi-decision-core",
    version="0.5.0",
    py_modules=["magi_system_v5"],
    install_requires=[
        "openai",
        "anthropic",
        "aiohttp",
        "tenacity",
        "sentence-transformers",
        "markdown",
        "pandas",
        "pytest",
        "pytest-asyncio",
        "python-dotenv",
        "torch",
        "rich"
    ],
    entry_points={"console_scripts": ["magi=magi_system_v5:main"]},
    author="Your Name",
    description="A multi-agent deliberation system inspired by Neon Genesis Evangelion's MAGI",
    url="https://github.com/yourusername/magi-decision-core"
)