# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'demo-tuner',
    version = '0.1',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.6',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: '
    ],
    author = 'Nan Shun',
    author_email = 'nan@sari.ac.cn',
    description = 'NNI control for Neural Network Intelligence project',
    license = 'MIT'
)
