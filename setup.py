from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ops-framework",
    version="1.0.0",
    author="Athanasios Karatzioulas",
    author_email="",
    description="Operational Projective Stability framework for multi-channel time series analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karatzioulasathanasios-creator/ops-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.6",
        "matplotlib>=3.3",
    ],
)