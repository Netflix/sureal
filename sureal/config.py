import os

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

PYTHON_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


class SurealConfig(object):

    @classmethod
    def root_path(cls, *components):
        return os.path.join(PYTHON_ROOT, *components)

    @classmethod
    def workspace_path(cls, *components):
        return cls.root_path('workspace', *components)

    @classmethod
    def workdir_path(cls, *components):
        return cls.root_path('workspace', 'workdir', *components)

    @classmethod
    def resource_path(cls, *components):
        return cls.root_path('resource', *components)

    @classmethod
    def test_resource_path(cls, *components):
        return cls.root_path('test', 'resource', *components)


class DisplayConfig(object):

    @staticmethod
    def show(**kwargs):
        import matplotlib.pyplot as plt
        if 'write_to_dir' in kwargs:
            format = kwargs['format'] if 'format' in kwargs else 'png'
            filedir = kwargs['write_to_dir'] if kwargs['write_to_dir'] is not None else SurealConfig.workspace_path('output')
            os.makedirs(filedir, exist_ok=True)
            for fignum in plt.get_fignums():
                fig = plt.figure(fignum)
                fig.savefig(os.path.join(filedir, str(fignum) + '.' + format), format=format)
        else:
            plt.show()
