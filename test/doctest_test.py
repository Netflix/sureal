__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

"""
Run embedded doctests
"""

import doctest

from sureal.tools import misc


def test_doctest():
    doctest.testmod(misc)
