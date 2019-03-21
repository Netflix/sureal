import sys
import time

import numpy as np
from scipy import linalg

from sureal.subjective_model import SubjectiveModel
from sureal.dataset_reader import PairedCompDatasetReader

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class PairedCompSubjectiveModel(SubjectiveModel):

    def _assert_args(self):
        super(PairedCompSubjectiveModel, self)._assert_args()
        assert isinstance(self.dataset_reader, PairedCompDatasetReader)

    @classmethod
    def from_dataset_file(cls, dataset_filepath, content_ids=None, asset_ids=None):
        dataset = cls._import_dataset_and_filter(dataset_filepath, content_ids, asset_ids)
        dataset_reader = PairedCompDatasetReader(dataset)
        return cls(dataset_reader)

    @staticmethod
    def _get_ref_mos(dataset_reader, mos):
        raise NotImplementedError

    @staticmethod
    def _get_opinion_score_2darray_with_preprocessing(dataset_reader, **kwargs):
        raise NotImplementedError


class BradleyTerryNewtonRaphsonPairedCompSubjectiveModel(PairedCompSubjectiveModel):
    """ Bradley-Terry model to convert paired comparison scores to continuous score.

    Implementation based on: http://sites.stat.psu.edu/~drh20/code/btmatlab/btnr.m

    function info=btnr(wm)
    % function info=btnr(wm)
    %
    % btnr uses a Newton-Raphson algorithm to fit the Bradley-Terry model.
    %
    % The input wm is an nxn matrix such that wm(i,j) is the number of times
    % team i beat team j.
    %
    % The output is an nx1 vector of parameter estimates of "team skill".
    %
    % This algorithm does not contain any checks for lack of convergence;
    % it is assumed that for any partition of the teams into sets A and B,
    % at least one team from A beats at least one team from B at least once.

    n=size(wm,1);
    %flops(0);

    % The parameter vector gamma is constrained to keep its last
    % entry equal to 0; thus, nmo stands for "n minus one".
    nmo=n-1;
    gamma=zeros(n,1); % initial value:  All teams equal
    iteration=0;
    change=realmax;
    gm=wm(1:nmo,:)+wm(:,1:nmo)';
    wins=sum(wm(1:nmo,:),2);
    gind=(gm>0);
    squaregind=gind(:,1:nmo);
    y=zeros(nmo,n);
    rus=y;

    while stats.norm(change) > 1e-08
        iteration=iteration+1;
        pi=exp(gamma);
        pius=pi(:,ones(n,1));
        piust=pius(:,1:nmo)';
        pius=pius(1:nmo,:);
        rus(gind)=pius(gind) ./ (pius(gind)+piust(gind));
        y(gind) = gm(gind) .* rus(gind);
        d2Q=-sum(y,2);
        dL=wins+d2Q;
        y(gind)=y(gind).*(1-rus(gind));
        d2L=-diag(sum(y,2));
        d2L(squaregind)=y(squaregind);
        A=d2L;

        cov = -inv(d2L);

        change=(A)\dL;
        %change = inv(d2L) * dL;
        change(n) = 0;
        gamma = gamma - change;
    end
    Iterations = iteration
    %Floating_point_operations=flops
    info=exp(gamma);
    info(n)=1;

    """

    TYPE = 'BTNR'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        wm = np.nansum(dataset_reader.opinion_score_3darray, axis=2)

        # wm = np.array(
        #     [[0, 3, 2, 7],
        #      [1, 0, 6, 3],
        #      [4, 3, 0, 0],
        #      [1, 2, 5, 0]]
        #               )

        n, m = wm.shape
        assert n == m

        nmo = n - 1
        gamma = np.zeros(n)
        iteration = 0
        change = sys.float_info.max
        gm = wm[0:nmo, :] + wm[:, 0:nmo].transpose()
        wins = np.sum(wm[0:nmo, :], axis=1)
        gind = (gm > 0)
        squaregind = gind[:, 0:nmo]
        y = np.zeros([nmo, n])
        rus = y.copy()
        cov = None

        DELTA_THR = 1e-8

        while linalg.norm(change) > DELTA_THR:
            iteration += 1
            pi = np.exp(gamma)
            pius = np.tile(pi, (n, 1)).transpose()
            piust = pius[:, 0:nmo].transpose()
            pius = pius[0:nmo, :]
            rus[gind] = np.divide(pius[gind], (pius[gind] + piust[gind]))
            y[gind] = np.multiply(gm[gind], rus[gind])
            d2Q = - np.sum(y, axis=1)
            dL = wins + d2Q
            y[gind] = np.multiply(y[gind], (1. - rus[gind]))
            d2L = -np.diag(np.sum(y, axis=1))
            d2L[squaregind] = y[:, :-1][squaregind]

            cov = - linalg.pinv(d2L)

            change = np.matmul(linalg.pinv(d2L), dL)
            change = np.hstack([change, np.array([0])])
            gamma -= change

            msg = 'Iteration {itr:4d}: change {change}, mean x_e {x_e}'.format(itr=iteration, change=linalg.norm(change), x_e=np.mean(gamma))
            sys.stdout.write(msg + '\r')
            sys.stdout.flush()
            time.sleep(0.1)

        # scores = np.exp(gamma)
        # scores[-1] = 1.
        # instead of original formulation, output non-exponential score
        scores = gamma
        scores[-1] = 0.0

        std = np.diagonal(cov)
        std = np.hstack([std, np.array([0])])

        zscore_output = kwargs['zscore_output'] if 'zscore_output' in kwargs and 'zscore_output' is not None else False

        if zscore_output:
            scores_mean = np.mean(scores)
            scores_std = np.std(scores)
            scores = (scores - scores_mean) / scores_std
            std = std / scores_std

        result = {'quality_scores': list(scores),
                  'quality_scores_std': list(std)}
        return result
