# this script allows us to publish sureal in PyPi and the use it using 'pip install sureal'
# run 'python setup.py sdist bdist_wheel' to create the distribution files within sureal/python/src/

import os
import setuptools
from numpy.distutils.core import setup

parent_dir = os.getcwd() + '/'
curr_dir = parent_dir + 'python/src/'
build_dest_dir = '/tmp/sureal_packaging_temp'

# hacky way to exclude config.py: create a folder udner /tmp with same structure, but without config.py
os.system('cp -r {curr_dir} {build_dest_dir}'.format(curr_dir=curr_dir, build_dest_dir=build_dest_dir))
os.system('rm {build_dest_dir}/sureal/config.py'.format(build_dest_dir=build_dest_dir))

# change dir so that setup.py builds distribution files on the files we want
os.chdir(build_dest_dir)

PACKAGE_NAME = 'sureal'

with open(parent_dir + "README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=PACKAGE_NAME,
    version="0.1.2",
    author="Zhi Li",
    author_email="zli@netflix.com",
    description="Subjective quality scores recovery from noisy measurements.",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.12.0', 'scipy>=0.17.1', 'matplotlib>=2.0.0', 'pandas>=0.19.2'],
    include_package_data=True,
    url="https://github.com/Netflix/sureal",
    classifiers=(
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
    ),
)
