import os
import json
import hashlib
import sys

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

def persist_to_file(file_name):
    """
    Cache (or persist) returned value of function in a json file .
    """

    def decorator(original_func):

        if not os.path.exists(file_name):
            cache = {}
        else:
            try:
                cache = json.load(open(file_name, 'rt'))
            except (IOError, ValueError):
                sys.exit(1)

        def new_func(*args):
            h = hashlib.sha1(str(original_func.__name__) + str(args)).hexdigest()
            if h not in cache:
                cache[h] = original_func(*args)
                json.dump(cache, open(file_name, 'wt'))
            return cache[h]

        return new_func

    return decorator
