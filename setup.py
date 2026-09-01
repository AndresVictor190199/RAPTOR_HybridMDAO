"""
Setup script for the HybridPropulsion_Raptor package.

Installation:
    pip install -e .               # Development (editable)
    pip install .                   # Standard install
"""
from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
long_description = ""
if os.path.exists(os.path.join(here, "README.md")):
    with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="hpraptor",
    version="0.3.0",
    author="Victor Berrazueta",
    author_email="victor.berrazueta@epn.edu.ec",
    description=(
        "Energy optimization for hybrid propulsion systems "
        "in Transition VTOL UAVs — built on the RAPTOR framework."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VAero-Lab/HybridPropulsion_Raptor",
    packages=find_packages(),
    package_data={"hpraptor": ["../data/**/*.json", "../data/**/*.csv", "../data/**/*.npz"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.5",
        "pyyaml>=6.0",
        "requests>=2.25",
        "aerosandbox>=4.2",
    ],
    extras_require={
        "mdao": ["openmdao>=3.30"],
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=5.0"],
        "all": ["openmdao>=3.30", "plotly>=5.0", "pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "uav", "vtol", "hybrid-propulsion", "optimization",
        "energy-model", "transition-vtol", "fuel-cell",
        "series-hybrid", "openmdao", "mdao",
    ],
    entry_points={
        "console_scripts": [
            "hpraptor-run=run_mission:main",
            "hpraptor-mdao=hpraptor_mdao.run:main",
            "hpraptor-dem=hpraptor.m1_mission.srtm_downloader:main",
            "hpraptor-3d=hpraptor.postprocessing.trajectory_3d:main",
            "hpraptor-serve=hpraptor.postprocessing.serve:main",
        ],
    },
)
