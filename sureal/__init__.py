from sureal.subjective_model import MosModel, SubjrejMosModel, \
    BiasremvSubjrejMosModel, SubjectMLEModelProjectionSolver2
from sureal.routine import run_subjective_models

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

# Note, version doesn't have to be hard-coded like this, it can also be
# determined programmatically
__version__ = "0.9.0"


class BT500Model(SubjrejMosModel):
    TYPE = 'BT500'


class P913124Model(BiasremvSubjrejMosModel):
    TYPE = 'P913'


class P910AnnexEModel(SubjectMLEModelProjectionSolver2):
    TYPE = 'P910'


__all__ = ['P910AnnexEModel', 'P913124Model', 'BT500Model', 'MosModel',
           'run_subjective_models']
