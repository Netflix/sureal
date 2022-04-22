import os
import subprocess
import time
import unittest
from time import sleep
import multiprocessing

import numpy as np

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


try:
    multiprocessing.set_start_method('fork')
except ValueError:  # noqa, If platform does not support, just ignore
    pass
except RuntimeError:  # noqa, If context has already being set, just ignore
    pass


def empty_object():
    return type('', (), {})()


def get_unique_sorted_list(l):
    """
    >>> get_unique_sorted_list([3, 4, 4, 1])
    [1, 3, 4]
    >>> get_unique_sorted_list([])
    []
    """
    return sorted(list(set(l)))


def get_file_name_without_extension(path):
    """

    >>> get_file_name_without_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01'
    >>> get_file_name_without_extension('yuv/src01_hrc01')
    'src01_hrc01'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.yuv')
    'src01_hrc01'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.sdr.yuv')
    'src01_hrc01.sdr'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.sdr.dvi.yuv')
    'src01_hrc01.sdr.dvi'

    """
    return os.path.splitext(path.split("/")[-1])[0]


def get_file_name_with_extension(path):
    """

    >>> get_file_name_with_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01.yuv'
    >>> get_file_name_with_extension('src01_hrc01.yuv')
    'src01_hrc01.yuv'
    >>> get_file_name_with_extension('abc/xyz/src01_hrc01.yuv')
    'src01_hrc01.yuv'

    """
    return path.split("/")[-1]


def get_file_name_extension(path):
    '''
    >>> get_file_name_extension("file:///mnt/zli/test.txt")
    'txt'
    >>> get_file_name_extension("test.txt")
    'txt'
    >>> get_file_name_extension("abc")
    'abc'
    '''
    return path.split('.')[-1]


def indices(a, func):
    """
    Get indices of elements in an array which satisfies func
    >>> indices([1, 2, 3, 4], lambda x: x>2)
    [2, 3]
    >>> indices([1, 2, 3, 4], lambda x: x==2.5)
    []
    >>> indices([1, 2, 3, 4], lambda x: x>1 and x<=3)
    [1, 2]
    >>> indices([1, 2, 3, 4], lambda x: x in [2, 4])
    [1, 3]
    """
    return [i for (i, val) in enumerate(a) if func(val)]


def import_json_file(filepath):
    """
    Import a JSON-formatted input file as a dict.
    :param filepath:
    :return:
    """
    import json
    from argparse import Namespace
    with open(filepath, 'r') as in_f:
        ret = json.load(in_f)
    ns = Namespace(**ret)  # convert dict to namespace
    return ns


def import_python_file(filepath):
    """
    Import a python file as a module.
    :param filepath:
    :return:
    """
    filename = get_file_name_without_extension(filepath)
    try:
        from importlib.machinery import SourceFileLoader
        ret = SourceFileLoader(filename, filepath).load_module()
    except ImportError:
        import imp
        ret = imp.load_source(filename, filepath)
    return ret


def get_cmd_option(argv, begin, end, option):
    '''

    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 3, 5, '--xyz')
    '123'
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 0, 5, '--xyz')
    '123'
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 4, 5, '--xyz')
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 5, 5, '--xyz')
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 6, 5, '--xyz')
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 0, 5, 'a')
    'b'
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 0, 5, 'b')
    'c'

    '''
    itr = None
    for itr in range(begin, end):
        if argv[itr] == option:
            break
    if itr is not None and itr != end and (itr + 1) != end:
        return argv[itr + 1]
    return None


def cmd_option_exists(argv, begin, end, option):
    '''

    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 2, 4, 'c')
    True
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 3, 4, 'c')
    False
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 3, 4, 'd')
    True
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 2, 4, 'a')
    False
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 2, 4, 'b')
    False

    '''
    found = False
    for itr in range(begin, end):
        if argv[itr] == option:
            found = True
            break
    return found


def parallel_map(func, list_args, processes=None, pause_sec=0.01):
    """
    Build my own parallelized map function since multiprocessing's Process(),
    or Pool.map() cannot meet my both needs:
    1) be able to control the maximum number of processes in parallel
    2) be able to take in non-picklable objects as arguments
    """

    # get maximum number of active processes that can be used
    max_active_procs = processes if processes is not None else multiprocessing.cpu_count()

    # create shared dictionary
    return_dict = multiprocessing.Manager().dict()

    # define runner function
    def func_wrapper(idx_args):
        idx, args = idx_args
        executor = func(args)
        return_dict[idx] = executor

    # add idx to args
    list_idx_args = []
    for idx, args in enumerate(list_args):
        list_idx_args.append((idx, args))

    procs = []
    for idx_args in list_idx_args:
        proc = multiprocessing.Process(target=func_wrapper, args=(idx_args,))
        procs.append(proc)

    waiting_procs = set(procs)
    active_procs = set([])

    # processing
    while True:

        # check if any procs in active_procs is done; if yes, remove them
        for p in active_procs.copy():
            if not p.is_alive():
                active_procs.remove(p)

        # check if can add a proc to active_procs (add gradually one per loop)
        if len(active_procs) < max_active_procs and len(waiting_procs) > 0:
            # move one proc from waiting_procs to active_procs
            p = waiting_procs.pop()
            active_procs.add(p)
            p.start()

        # if both waiting_procs and active_procs are empty, can terminate
        if len(waiting_procs) == 0 and len(active_procs) == 0:
            break

        if pause_sec is not None:
            sleep(pause_sec) # check every x sec

    # finally, collect results
    rets = list(map(lambda idx: return_dict[idx], range(len(list_args))))

    return rets


if __name__ == '__main__':
    import doctest
    doctest.testmod()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def weighed_nanmean_2d(a, weights, axis):
    assert len(a.shape) == 2
    assert axis in [0, 1]
    dim0, dim1 = a.shape
    if axis == 0:
        return np.divide(
            np.nansum(np.multiply(a,            np.tile(weights, (dim1, 1)).T), axis=0),
            np.nansum(np.multiply(~np.isnan(a), np.tile(weights, (dim1, 1)).T), axis=0)
        )
    elif axis == 1:
        return np.divide(
            np.nansum(np.multiply(a,            np.tile(weights, (dim0, 1))), axis=1),
            np.nansum(np.multiply(~np.isnan(a), np.tile(weights, (dim0, 1))), axis=1),
        )
    else:
        assert False


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.verificationErrors = []
        self.maxDiff = None

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        try:
            super().assertAlmostEqual(first, second, places, msg, delta)
        except AssertionError as e:
            self.verificationErrors.append(str(e))


def run_process(cmd, **kwargs):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, **kwargs)
    except subprocess.CalledProcessError as e:
        raise AssertionError(f'Process returned {e.returncode}, cmd: {cmd}, msg: {str(e.output)}')
    return 0


def cmap_factory(name):
    assert name in ['red2green', 'red2green2']
    if name == 'red2green':
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    elif name == 'red2green2':
        from matplotlib.colors import LinearSegmentedColormap
        c = ["darkred", "red", "lightcoral", "white",
             "palegreen", "green", "darkgreen"]
        v = [0, .15, .4, .5, 0.6, .9, 1.]
        ll = list(zip(v, c))
        cmap = LinearSegmentedColormap.from_list('rg', ll, N=256)
    else:
        assert False
    return cmap
