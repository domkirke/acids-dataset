import os
import subprocess
import setuptools

# imports __version__
exec(open('rave/version.py').read())


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

extra_requirements = {
    'after': ["pretty-midi"]
}

__VERSION__ =  get_latest_git_tag()


setuptools.setup(
    name="acids-dataset",
    version=__version__,  # type: ignore
    author="Nils Démerlé, Axel Chemla--Romeu-Santos",
    author_email="demerle@ircam.fr, chemla@ircam.fr",
    description="a pre-processed dataset library for generative audio ML",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": [
        "preprocess = cli:preprocess",
    ]},
    install_requires=requirements.split("\n"),
    python_requires='>=3.9',
    include_package_data=True,
)
