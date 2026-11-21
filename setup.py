#!/usr/bin/env python

# this script allows us to publish sureal in PyPi and the use it using 'pip install sureal'
# run 'python setup.py sdist bdist_wheel' to create the distribution files within sureal/python/src/

from setuptools import setup


setup(
    name="sureal",
    setup_requires=["setupmeta"],
    versioning="dev",
    author="Zhi Li",
    author_email="zli@netflix.com",
    include_package_data=True,
    url="https://github.com/Netflix/sureal",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        'Operating System :: Unix',
    ],
    entry_points={
        'console_scripts': [
            'sureal = sureal.__main__:main'
        ]
    },
)
