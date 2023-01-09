import os
from setuptools import setup


with open("requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

setup(
    name = "marqo_clip_onnx",
    version = "0.0.1",
    author = "marqo-Li Wan",
    install_requires = install_requires,
)