# this script allows us to publish sureal in PyPi and the use it using 'pip install sureal'
# run 'python setup.py sdist bdist_wheel' to create the distribution files within sureal/python/src/

import os
import setuptools
from numpy.distutils.core import setup

PACKAGE_NAME = 'sureal'

parent_dir = os.path.dirname(__file__)


def _version():
    """ Get the local package version.
    """
    path = os.path.join(PACKAGE_NAME, "version.py")
    namespace = {}
    with open(path) as stream:
        exec(stream.read(), namespace)
    return namespace["__version__"]


with open(os.path.join(parent_dir, "README.md"), "r") as fh:
    long_description = fh.read()

try:
    import pypandoc
    long_description = pypandoc.convert_text(long_description, 'rst', format='md')
except ImportError:
    print("pypandoc module not found, could not convert Markdown to RST")

setup(
    name=PACKAGE_NAME,
    version=_version(),
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS",
        'Operating System :: Unix',
    ),
    entry_points={
        'console_scripts': [
            'sureal = sureal.__main__:main'
        ]
    },
)
