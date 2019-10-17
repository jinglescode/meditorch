# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="meditorch",
    version="0.0.3",
    author="Jingles",
    author_email="jinglescode@gmail.com",
    description="A PyTorch package for biomedical image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jinglescode/meditorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
    project_urls={
        'Documentation': 'https://meditorch.readthedocs.io',
        'Source': 'https://github.com/jinglescode/meditorch',
        'Tracker': 'https://github.com/jinglescode/meditorch/issues',
    },
    install_requires=install_requires
)
