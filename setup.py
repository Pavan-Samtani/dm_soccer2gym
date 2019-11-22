import os
from setuptools import setup
from setuptools import find_packages

setup(
    name = "dm_soccer2gym",
    version = "0.0.4",
    author = "Martin Seiler",
    description = ("OpenAI Gym Wrapper for DeepMind Control Soccer"),
    license = "",
    keywords = "openai gym deepmind wrapper",
    packages=find_packages(),
    # install_requires=[
    #     'gym',
    #     'dm_control',
    # ],
)