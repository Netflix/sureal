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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
