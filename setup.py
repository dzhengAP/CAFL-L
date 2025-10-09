“”“Setup configuration for CAFL package.”””

from setuptools import setup, find_packages

with open(“README.md”, “r”, encoding=“utf-8”) as f:
long_description = f.read()

setup(
name=“cafl”,
version=“0.1.0”,
description=“Constraint-Aware Federated Learning with token-budget preservation”,
long_description=long_description,
long_description_content_type=“text/markdown”,
author=“Your Name”,
author_email=“your.email@example.com”,
url=“https://github.com/yourusername/cafl”,
packages=find_packages(where=“src”),
package_dir={””: “src”},
python_requires=”>=3.8”,
install_requires=[
“torch>=2.0.0”,
“numpy>=1.21.0”,
“matplotlib>=3.5.0”,
“requests>=2.28.0”,
],
extras_require={
“dev”: [
“pytest>=7.0.0”,
“black>=22.0.0”,
“flake8>=4.0.0”,
“mypy>=0.950”,
],
},
entry_points={
“console_scripts”: [
“cafl-train=cafl.train:main”,
],
},
classifiers=[
“Development Status :: 3 - Alpha”,
“Intended Audience :: Science/Research”,
“License :: OSI Approved :: MIT License”,
“Programming Language :: Python :: 3.8”,
“Programming Language :: Python :: 3.9”,
“Programming Language :: Python :: 3.10”,
“Topic :: Scientific/Engineering :: Artificial Intelligence”,
],
)