# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='meditorch',
    version='0.0.1',
    description='A PyTorch package for biomedical image processing',
    long_description=readme,
    author='Jingles',
    author_email='jinglescode@gmail.com',
    url='https://github.com/jinglescode/meditorch',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

