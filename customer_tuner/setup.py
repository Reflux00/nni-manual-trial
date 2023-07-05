# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'SAC-tuner',
    version = '0.1',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.6',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: '
    ],
    author = 'Nan Shun',
    author_email = 'nan@sari.ac.cn',
    description = 'SAC for Neural Network Intelligence project',
    license = 'MIT'
)
