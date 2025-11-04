#!/usr/bin/env python3
"""
Setup script for GPU-Accelerated Audio Visualization Application
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-visualizer",
    version="1.0.0",
    author="Audio Visualization Team",
    author_email="team@audiovisualization.com",
    description="GPU-accelerated interactive audio visualization with spectrogram, cepstrogram, and F-K transform analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/audio-visualization/audio-visualizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=11.0.0"],
        "gpu-cuda12": ["cupy-cuda12x>=12.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio-visualizer=audio_visualizer.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "audio_visualizer": [
            "rendering/shaders/*.glsl",
            "*.md",
        ],
    },
    keywords="audio visualization spectrogram cepstrogram fk-transform gpu cuda real-time",
    project_urls={
        "Bug Reports": "https://github.com/audio-visualization/audio-visualizer/issues",
        "Source": "https://github.com/audio-visualization/audio-visualizer",
        "Documentation": "https://audio-visualizer.readthedocs.io/",
    },
)