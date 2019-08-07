import io
from setuptools import find_packages, setup


description = (
    """Thin tensorflow wrapper for a subset of Cirq functionality.""")


# Read in requirements
requirements = open("./requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

__version__ = "0.010"

setup(
    name="cirq-tf",
    version=__version__,
    url="https://github.com/peterse/tf-cirq",
    author="Evan",
    install_requires=requirements,
    license="Apache 2",
)
