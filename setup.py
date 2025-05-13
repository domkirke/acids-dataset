import os
import re
import subprocess
import setuptools

# imports __version__
__VERSION__ = None
version_filepath = 'acids_dataset/version.py'
if os.path.isfile(version_filepath):
    exec(open(version_filepath).read())
    __VERSION__ = globals()['__version__']


def get_latest_git_tag():
    try:
        latest_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"]).strip().decode('utf-8')
        if latest_tag.startswith('v'): latest_tag = latest_tag[1:]
        return latest_tag
    except Exception as e:
        print(f"Could not retrieve the latest tag: {e}")
        return "0.0.0"  # Fallback version


with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

# extra_requirements = {
#     'after': ["pretty-midi"]
# }

__VERSION__ =  __VERSION__ or get_latest_git_tag()

from pathlib import Path
def get_config_packages():
    config_packages = []
    for r, d, f in os.walk(Path(__file__).parent / "acids_dataset" / "configs"):
        gin_files = list(filter(lambda x: os.path.splitext(x)[1] == ".gin", f))
        if len(gin_files) > 0: 
            config_packages.append("/".join(Path(r).parts))
    return config_packages

def get_config_package_data():
    config_packages = {}
    for r, d, f in os.walk(Path(__file__).parent / "acids_dataset" / "configs"):
        gin_files = list(filter(lambda x: os.path.splitext(x)[1] == ".gin", f))
        if len(gin_files) > 0: 
            
            config_packages["/".join(Path(r).parts)] = ['*.gin']
    return config_packages


setuptools.setup(
    name="acids-dataset",
    version=__VERSION__,  # type: ignore
    author="Axel Chemla--Romeu-Santos, Nils Demerlé, Antoine Caillon",
    author_email="chemla@ircam.fr,demerle@ircam.fr",
    description="a pre-processed dataset library for generative audio ML",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_data = get_config_package_data(),
    package_data = {"": ["configs/*.gin"], "acids_dataset.transforms.basic_pitch_torch": ['assets/*.pth']},
    entry_points={"console_scripts": [
        "acids-dataset = acids_dataset.cli:main",
    ]},
    install_requires=requirements.split("\n"),
    python_requires='>=3.11',
    include_package_data=True,
)
