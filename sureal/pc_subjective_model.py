import sys
import time
from functools import partial

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.stats import norm

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
    def _get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs):
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

        change=(A) \\ dL;
        %change = inv(d2L) * dL;
        change(n) = 0;
        gamma = gamma - change;
    end
    Iterations = iteration
    %Floating_point_operations=flops
    info=exp(gamma);
    info(n)=1;

    """

    TYPE = 'BT_NR'
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

            # cov = - linalg.pinv(d2L)

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

        # std = np.diagonal(cov)
        # std = np.hstack([std, np.array([0])])

        zscore_output = kwargs['zscore_output'] \
            if 'zscore_output' in kwargs and kwargs['zscore_output'] is not None else False

        if zscore_output:
            scores_mean = np.mean(scores)
            scores_std = np.std(scores)
            scores = (scores - scores_mean) / scores_std
            # std = std / scores_std

        result = {'quality_scores': list(scores),
                  'quality_scores_std': None}
        return result


class BradleyTerryMlePairedCompSubjectiveModel(PairedCompSubjectiveModel):
    """ Bradley-Terry model based on maximum likelihood estimation, classical version.
    """

    TYPE = 'BT_MLE'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        alpha = np.nansum(dataset_reader.opinion_score_3darray, axis=2)

        v, stdv_v, p, stdv_p, cova_v, cova_p = cls.resolve_model(alpha, **kwargs)

        return {'quality_scores': v,
                'quality_scores_std': stdv_v,
                'quality_scores_ci95': [list(1.95996 * np.array(stdv_v)), list(1.95996 * np.array(stdv_v))],
                'quality_scores_p': p,
                'quality_scores_p_std': stdv_p,
                'quality_scores_p_cov': cova_p,
                'quality_scores_v_cov': cova_v}

    @staticmethod
    def resolve_model(alpha, **more):

        display = more['display'] if 'display' in more else True
        assert isinstance(display, bool)

        # example: alpha is paired-comparison matrix
        # alpha = np.array(
        #     [[0, 3, 2, 7],
        #      [1, 0, 6, 3],
        #      [4, 3, 0, 0],
        #      [1, 2, 5, 0]]
        #     )
        M, M_ = alpha.shape
        assert M == M_

        n = alpha + alpha.T

        iteration = 0
        p = 1.0 / M * np.ones(M)
        change = sys.float_info.max

        DELTA_THR = 1e-8

        while change > DELTA_THR:
            iteration += 1
            p_prev = p

            # p_i = (sum_j alpha_ij) / (sum_j n_ij / (p_i + p_j))
            # note that if p = [1, 2, 3, 4] and M = 4, then
            # np.tile(p, (M, 1)).T creates patterns like
            # [
            #  [1, 1, 1, 1],
            #  [2, 2, 2, 2],
            #  [3, 3, 3, 3],
            #  [4, 4, 4, 4]
            #  ]
            pp = np.tile(p, (M, 1)).T + np.tile(p, (M, 1))

            p = np.sum(alpha, axis=1) / np.sum(n / pp, axis=1)  # summing over axis=1 marginalizes j

            p = p / np.sum(p)  # re-normalize to force p a prob. distribution

            change = linalg.norm(p - p_prev)

            if display:
                msg = 'Iteration {itr:4d}: change {change}, mean p {p}'.format(itr=iteration, change=linalg.norm(change), p=np.mean(p))
                sys.stdout.write(msg + '\r')
                sys.stdout.flush()
                time.sleep(0.001)

        # lambda_ii = sum_j -alpha_ij / p_i^2 + n_ij / (p_i + p_j)^2
        # lambda_ij = n_ij / (p_i + p_j)^2, i != j
        # H = [lambda_ij]
        # C = [[-H, 1], [1', 0]]^-1 of (M + 1) x (M + 1)
        # variance of p_i is then diag(C)[i].

        pp = np.tile(p, (M, 1)).T + np.tile(p, (M, 1))
        lbda_ii = np.sum(-alpha / np.tile(p, (M, 1)).T**2 + n / pp**2, axis=1)  # summing over axis=1 marginalizes j
        lbda_ij = n / pp*2
        lbda = lbda_ij + np.diag(lbda_ii)
        cova_p = np.linalg.pinv(
            np.vstack([np.hstack([-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])]))
        vari_p = np.diagonal(cova_p)[:-1]
        stdv_p = np.sqrt(vari_p)
        cova_p = cova_p[:-1, :-1]
        cova_v = cova_p / (np.expand_dims(p, axis=1) * (np.expand_dims(p, axis=1).T))

        v = np.log(p)
        stdv_v = stdv_p / p  # y = log(x) -> dy = 1/x * dx

        return list(v), list(stdv_v), list(p), list(stdv_p), cova_v, cova_p


class ThurstoneMlePairedCompSubjectiveModel(PairedCompSubjectiveModel):
    """ Thurstone model based on maximum likelihood estimation, classical version
    """

    TYPE = 'THURSTONE_MLE'
    # VERSION = '1.0'
    # VERSION = '1.1'  # fix sign nllf sign issue
    VERSION = '1.2'  # add confidence interval

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        # example:
        # alpha = np.array(
        #     [[0, 3, 2, 7],
        #      [1, 0, 6, 3],
        #      [4, 3, 0, 0],
        #      [1, 2, 5, 0]]
        #     )
        alpha = np.nansum(dataset_reader.opinion_score_3darray, axis=2)

        scores, std, cov = cls.resolve_model(alpha, **kwargs)

        zscore_output = kwargs['zscore_output'] \
            if 'zscore_output' in kwargs and kwargs['zscore_output'] is not None else False

        if zscore_output:
            scores_mean = np.mean(scores)
            scores_std = np.std(scores)
            scores = (scores - scores_mean) / scores_std
            std = std / scores_std

        result = {
            'quality_scores': scores,
            'quality_scores_std': std,
            'quality_scores_ci95': [list(1.95996 * std), list(1.95996 * std)],
        }
        return result

    @classmethod
    def resolve_model(cls, alpha, **more):

        use_simplified_lbda = more['use_simplified_lbda'] if 'use_simplified_lbda' in more else True
        assert isinstance(use_simplified_lbda, bool)

        M, M_ = alpha.shape
        assert M == M_
        nllf_partial = partial(cls.neg_log_likelihood_function, alpha=alpha)
        v0 = np.zeros(M)
        ret = minimize(nllf_partial, v0, method='SLSQP', jac='2-point',
                       options={'ftol': 1e-8, 'disp': True, 'maxiter': 1000})
        assert ret.success, "minimization is unsuccessful."
        v = ret.x

        if use_simplified_lbda:
            vi_m_vj = np.tile(v, (M, 1)).T - np.tile(v, (M, 1))
            phi_vi_m_vj = norm.cdf(vi_m_vj)
            f_vi_m_vj = norm.pdf(vi_m_vj)
            lbda_ii = np.sum(
                (alpha + alpha.T) * (-vi_m_vj * phi_vi_m_vj * f_vi_m_vj - f_vi_m_vj ** 2) / phi_vi_m_vj ** 2
                , axis=1)  # summing over axis=1 marginalizes j
            lbda_ij = -(
                (alpha + alpha.T) * (-vi_m_vj * phi_vi_m_vj * f_vi_m_vj - f_vi_m_vj ** 2) / phi_vi_m_vj ** 2
            )
        else:
            vi_m_vj = np.tile(v, (M, 1)).T - np.tile(v, (M, 1))
            phi_vi_m_vj = norm.cdf(vi_m_vj)
            f_vi_m_vj   = norm.pdf(vi_m_vj)
            d_vi_m_vj   = - vi_m_vj * norm.pdf(vi_m_vj)
            phi_vj_m_vi = - phi_vi_m_vj
            f_vj_m_vi   =   f_vi_m_vj
            d_vj_m_vi   = - d_vi_m_vj
            lbda_ii = np.sum(
                alpha   * (phi_vi_m_vj * d_vi_m_vj - f_vi_m_vj ** 2) / phi_vi_m_vj ** 2 +
                alpha.T * (phi_vj_m_vi * d_vj_m_vi - f_vj_m_vi ** 2) / phi_vj_m_vi ** 2
            , axis=1)  # summing over axis=1 marginalizes j

            lbda_ij = -(
                alpha   * (phi_vi_m_vj * d_vi_m_vj - f_vi_m_vj ** 2) / phi_vi_m_vj ** 2 +
                alpha.T * (phi_vj_m_vi * d_vj_m_vi - f_vj_m_vi ** 2) / phi_vj_m_vi ** 2
            )

        lbda = lbda_ij + np.diag(lbda_ii)
        cova = np.linalg.pinv(
            np.vstack([np.hstack([-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])]))
        vari = np.diagonal(cova)[:-1]
        stdv = np.sqrt(vari)
        cova = cova[:-1, :-1]

        return v, stdv, cova

    @staticmethod
    def neg_log_likelihood_function(v, alpha):
        # nllf(.) = - sum_i,j log(n_ij / alpha_ij) + alpha_ij * log phi (v_i - v_j) + alpha_ji * log phi (v_j - vi)
        # note that if p = [1, 2, 3, 4] and M = 4, then
        # np.tile(p, (M, 1)).T creates patterns like
        # [
        #  [1, 1, 1, 1],
        #  [2, 2, 2, 2],
        #  [3, 3, 3, 3],
        #  [4, 4, 4, 4]
        #  ]
        M = alpha.shape[0]
        epsilon = 1e-8 / M
        mtx = alpha * np.log(
            norm.cdf(
                (np.tile(v, (M, 1)).T - np.tile(v, (M, 1)))
            ) + epsilon
        ) + alpha.T * np.log(
            norm.cdf(
                (np.tile(v, (M, 1)) - np.tile(v, (M, 1)).T)
            ) + epsilon
        )
        return - np.sum(mtx)
