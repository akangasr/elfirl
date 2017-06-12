import os
from setuptools import setup, find_packages
from io import open

packages = ['elfirl'] + ['elfirl.' + p for p in find_packages('elfirl')]

#with open('requirements.txt', 'r') as f:
#    requirements = f.read().splitlines()

setup(
    name='elfirl',
    packages=packages,
    version=0.1,
    author='Antti Kangasrääsiö',
    author_email='antti.kangasraasio@iki.fi',
    url='https://github.com/akangasr/elfirl',
#    install_requires=requirements,
    description='ELFI RL framework',
    license='MIT')
