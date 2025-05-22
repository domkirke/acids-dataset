import os
from setuptools import setup

version = os.environ.get("ACIDS_DATASET_VERSION")

setup(
    version=version,
)