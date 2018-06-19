import os

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

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
    import imp
    filename = get_file_name_without_extension(filepath)
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
