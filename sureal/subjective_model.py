import copy
from abc import ABCMeta, abstractmethod
import sys
import time

import numpy as np
from scipy import linalg
from scipy import stats
import pandas as pd
from scipy.stats import chi2, norm

from sureal.core.mixin import TypeVersionEnabled
from sureal.tools.misc import import_python_file, indices, weighed_nanmean_2d
from sureal.dataset_reader import RawDatasetReader
from sureal.tools.stats import vectorized_gaussian, vectorized_convolution_of_two_logistics, \
    vectorized_convolution_of_two_uniforms

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

__metaclass__ = ABCMeta


class SubjectiveModel(TypeVersionEnabled):
    """
    Base class for any model that takes the input of a subjective quality test
    experiment dataset with raw scores (dis_video must has key of 'os' (opinion
    score)) and output estimated quality for each impaired video (e.g. MOS, DMOS
    or more advanced estimate of subjective quality).

    A number of common functionalities are included: dscore_mode, zscore_mode,
     normalize_final, transform_final, subject_rejection
    """

    @classmethod
    @abstractmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        raise NotImplementedError

    def _assert_args(self):
        assert isinstance(self.dataset_reader, RawDatasetReader)

    def __init__(self, dataset_reader):
        TypeVersionEnabled.__init__(self)
        self.dataset_reader = dataset_reader
        self._assert_args()

    @classmethod
    def _import_dataset_and_filter(cls, dataset_filepath, content_ids, asset_ids):
        dataset = import_python_file(dataset_filepath)
        if content_ids is not None:
            dataset.dis_videos = [dis_video for dis_video in dataset.dis_videos if dis_video['content_id'] in content_ids]
        if asset_ids is not None:
            dataset.dis_videos = [dis_video for dis_video in dataset.dis_videos if dis_video['asset_id'] in asset_ids]
        return dataset

    @classmethod
    def from_dataset_file(cls, dataset_filepath, content_ids=None, asset_ids=None):
        dataset = cls._import_dataset_and_filter(dataset_filepath, content_ids, asset_ids)
        dataset_reader = RawDatasetReader(dataset)
        return cls(dataset_reader)

    def run_modeling(self, **kwargs):
        model_result = self._run_modeling(self.dataset_reader, **kwargs)

        try:
            observers = self.dataset_reader._get_list_observers()  # may not exist
            model_result['observers'] = observers
        except AssertionError:
            pass
        except AttributeError:
            pass

        self._postprocess_model_result(model_result, **kwargs)
        self.model_result = model_result
        return model_result

    def to_aggregated_dataset(self, **kwargs):
        self._assert_modeled()
        return self.dataset_reader.to_aggregated_dataset(
            self.model_result['quality_scores'],
            scores_std = self.model_result['quality_scores_std'] if 'quality_scores_std' in self.model_result else None,
            **kwargs)

    def to_aggregated_dataset_file(self, dataset_filepath, **kwargs):
        self._assert_modeled()
        self.dataset_reader.to_aggregated_dataset_file(
            dataset_filepath,
            self.model_result['quality_scores'],
            scores_std = self.model_result['quality_scores_std'] if 'quality_scores_std' in self.model_result else None,
            **kwargs)

    def _assert_modeled(self):
        assert hasattr(self, 'model_result'), \
            "self.model_result doesn't exist. Run run_modeling() first."
        assert 'quality_scores' in self.model_result, \
            "self.model_result must have quality_scores."

    @staticmethod
    def _get_ref_mos(dataset_reader, mos):
        ref_mos = []
        for dis_video in dataset_reader.dataset.dis_videos:
            # get the dis video's ref video's mos
            curr_content_id = dis_video['content_id']
            ref_indices = indices(
                list(zip(dataset_reader.content_id_of_dis_videos,
                    dataset_reader.disvideo_is_refvideo)),
                lambda content_id_is_refvideo:
                content_id_is_refvideo[1] and content_id_is_refvideo[0] == curr_content_id
            )
            assert len(ref_indices) == 1, \
                'Should have only one ref video for a dis video, ' \
                'but got {}'.format(len(ref_indices))
            ref_idx = ref_indices[0]

            ref_mos.append(mos[ref_idx])
        return np.array(ref_mos)

    @staticmethod
    def _stack_repetitions_along_axis(s_esr, axis):
        """
            Take the 3D input matrix, slice it along the 3rd axis and stack the resulting 2D matrices
            along the selected matrix while maintaining the correct order.
            :param s_esr: 3D array of the shape [E, S, R]
            :param axis: 0 or 1
            :return: 2D array containing the values
                - if axis=0, the new shape is [R*E, S]
                - if axis = 1, the new shape is [E, R*S]
        """

        assert len(s_esr.shape) == 3
        E, S, R = s_esr.shape

        if axis == 0:
            o = np.zeros([R * E, S])

            for r in range(R):
                o[r * E:(r + 1) * E, :] = s_esr[:, :, r]

        elif axis == 1:
            o = np.zeros([E, R * S])

            for r in range(R):
                o[:, r * S:(r + 1) * S] = s_esr[:, :, r]

        else:
            assert False

        return o

    @staticmethod
    def _get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs):

        s_esr = dataset_reader.opinion_score_3darray

        original_opinion_score_3darray = copy.deepcopy(s_esr)

        ret = dict()

        # dscore_mode: True - do differential-scoring
        #              False - don't do differential-scoring
        dscore_mode = kwargs['dscore_mode'] if 'dscore_mode' in kwargs else False

        # zscore_mode: True - do z-scoring (normalizing to 0-mean 1-std)
        #              False - don't do z-scoring
        zscore_mode = kwargs['zscore_mode'] if 'zscore_mode' in kwargs else False

        # bias_offset: True - do bias offset according to ITU-T P.913
        #              False - don't do bias offset
        bias_offset = kwargs['bias_offset'] if 'bias_offset' in kwargs else False

        # subject_rejection: True - do subject rejection
        #              False - don't do subject rejection
        subject_rejection = kwargs['subject_rejection'] if 'subject_rejection' in kwargs else False

        subject_rejection_type = kwargs['subject_rejection_type'] if 'subject_rejection_type' in kwargs else None

        if subject_rejection is False:
            assert subject_rejection_type is None, "subject_rejection must be True if " \
                                                   "subject_rejection_type is specified"
        else:
            if subject_rejection_type is None:
                subject_rejection_type = 'kurtosis'
            assert subject_rejection_type in ['kurtosis', 'pearson', 'spearman']


        assert not (zscore_mode is True and bias_offset is True)

        if dscore_mode is True:

            # make sure dataset has ref_score
            assert dataset_reader.dataset.ref_score is not None, \
                "For differential score, dataset must have attribute ref_score."

            E, S, R = s_esr.shape
            s_e = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(s_esr, axis=1), axis=1)  # mean along s
            s_e_ref = SubjectiveModel._get_ref_mos(dataset_reader, s_e)
            s_esr = s_esr + dataset_reader.ref_score - np.tile(s_e_ref, (S, 1)).T[:, :, None]

        if zscore_mode is True:
            E, S, R = s_esr.shape
            mu_s = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(s_esr, axis=0), axis=0)  # mean along e
            simga_s = np.nanstd(SubjectiveModel._stack_repetitions_along_axis(s_esr, axis=0), ddof=1, axis=0)  # std along e
            s_esr = (s_esr - np.tile(mu_s, (E, 1))[:, :, None]) / np.tile(simga_s, (E, 1))[:, :, None]

        if bias_offset is True:
            E, S, R = s_esr.shape

            # video-by-video, estimate MOS by averageing over subjects
            s_e = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(s_esr, axis=1), axis=1)  # mean along s

            # subject by subject, estimate subject bias by comparing
            # against MOS
            delta_es = s_esr - np.tile(s_e, (S, 1)).T[:, :, None]
            delta_s = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(delta_es, axis=0), axis=0)  # mean along e

            # remove bias from opinion scores
            s_esr = s_esr - np.tile(delta_s, (E, 1))[:, :, None]

            ret['bias_offset_estimate'] = delta_s

        if subject_rejection is True:
            if subject_rejection_type == 'kurtosis':
                E, S, R = s_esr.shape

                ps = np.zeros(S)
                qs = np.zeros(S)

                for s_e in SubjectiveModel._stack_repetitions_along_axis(s_esr, axis=1):
                    s_e_notnan = s_e[~np.isnan(s_e)]
                    mu = np.mean(s_e_notnan)
                    sigma = np.std(s_e_notnan)
                    kurt = stats.kurtosis(s_e_notnan, fisher=False)

                    if 2 <= kurt and kurt <= 4:
                        for idx_s, s in enumerate(s_e):
                            if not np.isnan(s):
                                if s >= mu + 2 * sigma:
                                    ps[np.mod(idx_s, S)] += 1  # mod to assign repetitions to the correct subject
                                if s <= mu - 2 * sigma:
                                    qs[np.mod(idx_s, S)] += 1
                    else:
                        for idx_s, s in enumerate(s_e):
                            if not np.isnan(s):
                                if s >= mu + np.sqrt(20) * sigma:
                                    ps[np.mod(idx_s, S)] += 1
                                if s <= mu - np.sqrt(20) * sigma:
                                    qs[np.mod(idx_s, S)] += 1
                rejections = []
                acceptions = []
                reject_1st_stats = []
                reject_2nd_stats = []
                for idx_s, subject in zip(list(range(S)), list(range(S))):
                    reject_1st_stat = (ps[idx_s] + qs[idx_s]) / (E * R)
                    reject_2nd_stat = np.abs((ps[idx_s] - qs[idx_s]) / (ps[idx_s] + qs[idx_s]))
                    reject_1st_stats.append(reject_1st_stat)
                    reject_2nd_stats.append(reject_2nd_stat)
                    if reject_1st_stat > 0.05 and reject_2nd_stat < 0.3:
                        rejections.append(subject)
                    else:
                        acceptions.append(subject)

                # if all of the subjects would be rejected, none will
                if len(rejections) == S:
                    for idx_rej in range(S):
                        acceptions.append(rejections[idx_rej])
                    rejections = []

                s_esr = s_esr[:, acceptions, :]

                observer_rejected = [False for _ in range(S)]
                for rejection_idx in rejections:
                    observer_rejected[rejection_idx] = True

                ret['observer_rejected'] = observer_rejected
                ret['observer_rejected_1st_stats'] = reject_1st_stats
                ret['observer_rejected_2nd_stats'] = reject_2nd_stats

            elif subject_rejection_type in ['pearson', 'spearman']:
                s_es = np.nanmean(s_esr,axis=2)  # mean over repetitions to have the same number of scores as MOSs
                # each column is an observer, row is a video
                mos = np.nanmean(s_es, axis=1)
                std_dev = np.nanstd(s_es, axis=1)

                # calculate correlation r of observers' scores with overall mos
                r = np.zeros(np.size(s_es, 1))
                for obs_idx in range(np.size(s_es,1)):
                    nans = np.isnan(s_es[:, obs_idx])
                    if subject_rejection_type == 'pearson':
                        r[obs_idx], _ = stats.pearsonr(s_es[:, obs_idx][~nans], mos[~nans])
                    elif subject_rejection_type == 'spearman':
                        r[obs_idx], _ = stats.spearmanr(s_es[:, obs_idx][~nans], mos[~nans])
                    else:
                        assert False

                # find the rejection threshold
                if (np.mean(r) - np.std(r)) > 0.7:
                    rejection_thr = 0.7
                else:
                    rejection_thr = np.mean(r) - np.std(r)

                s_esr = s_esr[:, r>rejection_thr, :]

                ret['observer_rejected'] = r <= rejection_thr
                ret['observer_rejected_1st_stats'] = r
                ret['observer_rejected_2nd_stats'] = np.zeros(np.size(r))

            else:
                assert False

        ret['opinion_score_3darray'] = s_esr
        ret['original_opinion_score_3darray'] = original_opinion_score_3darray

        return ret

    @staticmethod
    def _postprocess_model_result(result, **kwargs):

        # normalize_final: True - do normalization on final quality score
        #                  False - don't do
        normalize_final = kwargs['normalize_final'] if 'normalize_final' in kwargs else False

        # transform_final: True - do (linear or other) transform on final quality score
        #                  False - don't do
        transform_final = kwargs['transform_final'] if 'transform_final' in kwargs else None

        assert 'quality_scores' in result

        if normalize_final is False:
            pass
        else:
            quality_scores = np.array(result['quality_scores'])
            quality_scores = (quality_scores - np.mean(quality_scores)) / \
                             np.std(quality_scores)
            result['quality_scores'] = list(quality_scores)

        if transform_final is None:
            pass
        else:
            quality_scores = np.array(result['quality_scores'])
            output_scores = np.zeros(quality_scores.shape)
            if 'p2' in transform_final:
                # polynomial coef of order 2
                output_scores += transform_final['p2'] * quality_scores * quality_scores
            if 'p1' in transform_final:
                # polynomial coef of order 1
                output_scores += transform_final['p1'] * quality_scores
            if 'p0' in transform_final:
                # polynomial coef of order 0
                output_scores += transform_final['p0']
            result['quality_scores'] = list(output_scores)


class MosModel(SubjectiveModel):
    """
    Mean Opinion Score (MOS) subjective model.
    """
    TYPE = 'MOS'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        os_3darray = ret['opinion_score_3darray']
        original_os_3darray = ret['original_opinion_score_3darray']
        result = cls._get_mos_and_stats(os_3darray, original_os_3darray)
        if 'observer_rejected' in ret:
            result['observer_rejected'] = ret['observer_rejected']
            assert 'observer_rejected_1st_stats' in ret
            assert 'observer_rejected_2nd_stats' in ret
            result['observer_rejected_1st_stats'] = ret['observer_rejected_1st_stats']
            result['observer_rejected_2nd_stats'] = ret['observer_rejected_2nd_stats']

        return result

    @classmethod
    def _get_mos_and_stats(cls, os_3darray, original_os_3darray):
        mos = np.nanmean(MosModel._stack_repetitions_along_axis(os_3darray, axis=1), axis=1)  # mean along s, ignore NaN
        std = np.nanstd(MosModel._stack_repetitions_along_axis(os_3darray, axis=1), axis=1, ddof=1)  # sample std -- use ddof 1
        mos_std = std / np.sqrt(
            np.nansum(~np.isnan(MosModel._stack_repetitions_along_axis(os_3darray, axis=1)), axis=1)
        )  # std / sqrt(N), ignoring NaN
        result = {'quality_scores': list(mos),
                  'quality_scores_std': list(mos_std),
                  'quality_scores_ci95': [list(1.95996 * mos_std), list(1.95996 * mos_std)],
                  'quality_ambiguity': list(std),
                  'raw_scores': os_3darray,
                  }
        num_pvs, num_obs, max_reps = os_3darray.shape
        num_os = np.sum(~np.isnan(os_3darray))

        result['reconstructions'] = cls._get_reconstructions(mos, num_obs, max_reps)

        original_num_pvs, original_num_obs, original_max_reps = original_os_3darray.shape
        original_num_os = np.sum(~np.isnan(original_os_3darray))
        dof = cls._get_dof(original_num_pvs, original_num_obs, original_max_reps) / original_num_os  # dof per observation
        result['dof'] = dof

        loglikelihood = np.nansum(np.log(vectorized_gaussian(
            os_3darray,
            np.tile(mos, (num_obs, 1)).T[:, :, None],
            np.tile(std, (num_obs, 1)).T[:, :, None],
        ))) / num_os  # log-likelihood per observation
        result['loglikelihood'] = loglikelihood

        aic = 2 * dof - 2 * loglikelihood  # aic per observation
        result['aic'] = aic

        bic = np.log(original_num_os) * dof - 2 * loglikelihood  # bic per observation
        result['bic'] = bic

        return result

    @classmethod
    def _get_reconstructions(cls, x_e, S, R):
        x_esr_hat = np.zeros([len(x_e), S, R])
        for r in range(R):
            x_esr_hat[:, :, r] = np.tile(x_e, (S, 1)).T
        return x_esr_hat

    @classmethod
    def _get_dof(cls, E, S, R):
        return E * 2


class DmosModel(MosModel):
    """
    Differential Mean Opinion Score (DMOS) subjective model.
    Use the formula:
    DMOS = MOS + ref_score (e.g. 5.0) - MOS_of_ref_video
    """
    TYPE = 'DMOS'
    VERSION = '1.0'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, '{} is already doing dscoring, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        return super(DmosModel, self).run_modeling(**kwargs2)


class LiveDmosModel(SubjectiveModel):
    """
    Differential Mean Opinion Score (DMOS) subjective model based on:
    Study of Subjective and Objective Quality Assessment of Video,
    K. Seshadrinathan, R. Soundararajan, A. C. Bovik and L. K. Cormack,
    IEEE Trans. Image Processing, Vol. 19, No. 6, June 2010.

    Difference is:
    DMOS = MOS + ref_score (e.g. 5.0) - MOS_of_ref_video
    instead of
    DMOS = MOS_of_ref_video - MOS
    """
    TYPE = 'LIVE_DMOS'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, '{} is already doing dscoring, no need to repeat.'.format(cls.__class__.__name__)

        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, '{} is already doing zscoring, no need to repeat.'.format(cls.__class__.__name__)

        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        kwargs2['zscore_mode'] = True

        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs2)
        s_esr = ret['opinion_score_3darray']

        s_esr = (s_esr + 3.0) * 100.0 / 6.0

        score = np.nanmean(LiveDmosModel._stack_repetitions_along_axis(s_esr, axis=1), axis=1)  # mean along s
        result = {
            'quality_scores': score
        }
        return result


class LeastSquaresModel(SubjectiveModel):
    """
    Simple model considering:
    z_e,s = q_e + b_s
    Solve by forming linear systems and find least squares solution
    can recover q_e and b_s
    """
    TYPE = 'LS'
    VERSION = '0.1'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjectAwareGenerativeModel must not and need not ' \
                          'apply subject rejection.'

        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        score_mtx = ret['opinion_score_3darray']

        assert np.shape(score_mtx)[2] == 1, 'LeastSquareModel currently not supported for repeated votes'
        score_mtx = score_mtx[:, :, 0]  # TODO: add repetitions

        num_video, num_subject = score_mtx.shape

        A = np.zeros([num_video * num_subject, num_video + num_subject])
        for idx_video in range(num_video):
            for idx_subject in range(num_subject):
                cur_row = idx_video * num_subject + idx_subject
                A[cur_row][idx_subject] = 1.0
                A[cur_row][num_subject + idx_video] = 1.0
        y = np.array(score_mtx.ravel())

        # add the extra constraint that the first ref video has score MOS
        mos = pd.DataFrame(score_mtx).mean(axis=1)
        row = np.zeros(num_subject + num_video)
        row[num_subject + 0] = 1
        score = mos[0]
        A = np.vstack([A, row])
        y = np.hstack([y, [score]])

        b_q = np.dot(linalg.pinv(A), y)
        b = b_q[:num_subject]
        q = b_q[num_subject:]

        result = {
            'quality_scores': list(q),
            'observer_bias': list(b),
        }
        return result


class LegacyMaximumLikelihoodEstimationModel(SubjectiveModel):
    """
    Generative model that considers individual subject (or observer)'s bias and
    inconsistency. The observed score is modeled by:
    X_e,s = x_e + B_e,s
    where x_e is the true quality of distorted video e, and B_e,s ~ N(b_s, v_s)
    is the term representing subject s's bias (b_s) and inconsistency (v_s).
    The model is then solved via maximum likelihood estimation using belief
    propagation.

    Note: Similar to MaximumLikelihoodEstimationModelContentOblivious, except
    that it does not deal with missing data etc. (Early implmentation)
    """

    TYPE = "MLE_legacy"
    VERSION = '0.1'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        def one_or_nan(x):
            y = np.ones(x.shape)
            y[np.isnan(x)] = float('nan')
            return y

        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, 'SubjectAwareGenerativeModel must not and need not ' \
                          'apply subject rejection.'

        force_subjbias_zeromean = kwargs['force_subjbias_zeromean'] \
            if 'force_subjbias_zeromean' in kwargs and kwargs['force_subjbias_zeromean'] is not None else True
        assert isinstance(force_subjbias_zeromean, bool)

        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        x_es = ret['opinion_score_3darray']

        assert np.shape(x_es)[2] == 1, 'LegacyMaximumLikelihoodEstimationModel currently not supported for repeated votes'
        x_es = x_es[:, :, 0]  # TODO: add repetitions

        E, S = x_es.shape

        use_log = kwargs['use_log'] if 'use_log' in kwargs else False

        # === initialization ===

        mos = pd.DataFrame(x_es).mean(axis=1)

        x_e = mos # use MOS as initial value for x_e
        b_s = np.zeros(S)

        r_es = x_es - np.tile(x_e, (S, 1)).T # r_es: residual at e, s
        v_s = np.array(pd.DataFrame(r_es).std(axis=0, ddof=0))

        log_v_s = np.log(v_s)

        # === iteration ===

        MAX_ITR = 5000
        REFRESH_RATE = 0.1
        DELTA_THR = 1e-8

        print('=== Belief Propagation ===')

        itr = 0
        while True:

            x_e_prev = x_e

            # (8) b_s
            num = pd.DataFrame(x_es - np.tile(x_e, (S, 1)).T).sum(axis=0) # sum over e
            den = pd.DataFrame(one_or_nan(x_es)).sum(axis=0) # sum over e
            b_s_new = num / den
            b_s = b_s * (1.0 - REFRESH_RATE) + b_s_new * REFRESH_RATE

            a_es = x_es - np.tile(x_e, (S, 1)).T - np.tile(b_s, (E, 1))
            if use_log:
                # (9') log_v_s
                num = pd.DataFrame(-np.ones([E, S]) + a_es**2 / np.tile(v_s**2, (E, 1))).sum(axis=0) # sum over e
                den = pd.DataFrame(-2 * a_es**2 / np.tile(v_s**2, (E, 1))).sum(axis=0) # sum over e
                log_v_s_new = log_v_s - num / den
                log_v_s = log_v_s * (1.0 - REFRESH_RATE) + log_v_s_new * REFRESH_RATE
                v_s = np.exp(log_v_s)
            else:
                # (9) v_s
                num = pd.DataFrame(2 * np.ones([E, S]) * np.tile(v_s**3, (E, 1)) - 4 * np.tile(v_s, (E, 1)) * a_es**2).sum(axis=0) # sum over e
                den = pd.DataFrame(np.ones([E, S]) * np.tile(v_s**2, (E, 1)) - 3 * a_es**2).sum(axis=0) # sum over e
                v_s_new = num / den
                v_s = v_s * (1.0 - REFRESH_RATE) + v_s_new * REFRESH_RATE
                # v_s = np.maximum(v_s, np.zeros(v_s.shape))

            # (7) x_e
            num = pd.DataFrame((x_es - np.tile(b_s, (E, 1))) / np.tile(v_s**2, (E, 1))).sum(axis=1) # sum along s
            den = pd.DataFrame(one_or_nan(x_es) / np.tile(v_s**2, (E, 1))).sum(axis=1) # sum along s
            x_e_new = num / den
            x_e = x_e * (1.0 - REFRESH_RATE) + x_e_new * REFRESH_RATE

            itr += 1

            delta_x_e = linalg.norm(x_e_prev - x_e)

            msg = 'Iteration {itr:4d}: change {delta_x_e}, mean x_e {x_e}, mean b_s {b_s}, mean v_s {v_s}'.\
                format(itr=itr, delta_x_e=delta_x_e, x_e=np.mean(x_e), b_s=np.mean(b_s), v_s=np.mean(v_s))
            sys.stdout.write(msg + '\r')
            sys.stdout.flush()
            # time.sleep(0.001)

            if delta_x_e < DELTA_THR:
                break

            if itr >= MAX_ITR:
                break

        sys.stdout.write("\n")

        if force_subjbias_zeromean:
            mean_b_s = np.mean(b_s)
            b_s -= mean_b_s
            x_e += mean_b_s

        result = {
            'quality_scores': list(x_e),
            'observer_bias': list(b_s),
            'observer_inconsistency': list(v_s),
        }

        return result


class MaximumLikelihoodEstimationModel(SubjectiveModel):
    """
    Generative model that considers individual subjective (or observer)'s bias
    and inconsistency, as well as content's bias and ambiguity.
    The observed score in each repetition is modeled by:
    X_e,s,r = x_e + B_e,s,r + A_e,s,r
    where x_e is the true quality of distorted video e, and B_e,s,r ~ N(b_s, v_s)
    is the term representing observer s's bias (b_s) and inconsistency (v_s).
    A_e,s,r ~ N(0, a_c), where c is a function of e, or c = c(e), represents
    content c's ambiguity (a_c). The model is then solved via maximum
    likelihood estimation using belief propagation.
    """

    TYPE = 'MLE' # maximum likelihood estimation
    # VERSION = '0.1'
    VERSION = '0.2' # added confidence interval for parameters

    mode = 'DEFAULT'

    DEFAULT_GRADIENT_METHOD = 'simplified'
    DEFAULT_NUMERICAL_PDF = 'gaussian'
    DEFAULT_DELTA_THR = 1e-8
    DEFAULT_FORCE_SUBJBIAS_ZEROMEAN = True

    @staticmethod
    def loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, content_id_of_dis_videos, axis, numerical_pdf):
        E, S, R = x_esr.shape

        if numerical_pdf == 'gaussian':
            # solution not unique
            a_c_e = np.array([a_c[i] for i in content_id_of_dis_videos])
            mu_esr = np.tile(x_e, (S, 1)).T[:, :, None] + np.tile(b_s, (E, 1))[:, :, None]
            vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
            ret = np.log(vectorized_gaussian(x_esr, mu_esr, np.sqrt(vs2_add_ace2)))

        elif numerical_pdf == 'logistic':
            a_c_e = np.array([a_c[i] for i in content_id_of_dis_videos])
            mu1 = np.tile(b_s, (E, 1))[:, :, None]
            mu2 = np.tile(x_e, (S, 1)).T[:, :, None]
            s1 = np.sqrt(np.tile((v_s / (np.pi / np.sqrt(3.0)))**2, (E, 1)))[:, :, None]
            s2 = np.sqrt(np.tile((a_c_e / (np.pi / np.sqrt(3.0)))**2, (S, 1)).T)[:, :, None]
            # ret = np.log(vectorized_logistic(x_es, mu1 + mu2, np.sqrt(s1**2 + s2**2)))
            ret = np.log(vectorized_convolution_of_two_logistics(x_esr, mu1, s1, mu2, s2))

        elif numerical_pdf == 'uniform':
            # gradient descent won't work due to zero density
            a_c_e = np.array([a_c[i] for i in content_id_of_dis_videos])
            mu1 = np.tile(b_s, (E, 1))[:, :, None]
            mu2 = np.tile(x_e, (S, 1)).T[:, :, None]
            s1 = np.sqrt(np.tile(v_s**2, (E, 1)) * 12.)[:, :, None]
            s2 = np.sqrt(np.tile(a_c_e**2, (S, 1)).T * 12.)[:, :, None]
            ret = np.log(vectorized_convolution_of_two_uniforms(x_esr, mu1, s1, mu2, s2))
        else:
            assert False, 'Unknown numerical_pdf: {}'.format(numerical_pdf)

        ret = np.nansum(MaximumLikelihoodEstimationModel._stack_repetitions_along_axis(ret, axis=axis), axis=axis)
        return ret

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        # mode: DEFAULT - full model
        #       SUBJECT_OBLIVIOUS - model not considering subject bias and inconsistency
        #       CONTENT_OBLIVIOUS - model not considering content ambiguity

        assert cls.mode in ['DEFAULT', 'SUBJECT_OBLIVIOUS', 'CONTENT_OBLIVIOUS']

        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} must not and need not apply subject rejection.'.format(cls.__name__)

        gradient_method = kwargs['gradient_method'] if 'gradient_method' in kwargs else cls.DEFAULT_GRADIENT_METHOD
        assert gradient_method == 'simplified' or gradient_method == 'original' or gradient_method == 'numerical'

        numerical_pdf = kwargs['numerical_pdf'] if 'numerical_pdf' in kwargs else cls.DEFAULT_NUMERICAL_PDF

        delta_thr = kwargs['delta_thr'] if 'delta_thr' in kwargs else cls.DEFAULT_DELTA_THR

        force_subjbias_zeromean = kwargs['force_subjbias_zeromean'] \
            if 'force_subjbias_zeromean' in kwargs and kwargs['force_subjbias_zeromean'] is not None \
            else cls.DEFAULT_FORCE_SUBJBIAS_ZEROMEAN
        assert isinstance(force_subjbias_zeromean, bool)

        def sum_over_content_id(xs, cids, num_c):
            assert len(xs) == len(cids)
            for cid in set(cids):
                assert cid in range(num_c), \
                    'content id must be in [0, {num_c}), but is {cid}'.format(num_c=num_c, cid=cid)
            sums = np.zeros(num_c)
            for x, cid in zip(xs, cids):
                sums[cid] += x
            return sums

        def std_over_subject_and_content_id(x_esr, cids, num_c):
            assert x_esr.shape[0] == len(cids)
            for cid in set(cids):
                assert cid in range(num_c), \
                    'content id must be in [0, {num_c}), but is {cid}'.format(num_c=num_c, cid=cid)
            ls = [[] for _ in range(num_c)]
            for idx_cid, cid in enumerate(cids):
                ls[cid] = ls[cid] + list(x_esr[idx_cid, :, :].ravel())
            stds = []
            for l in ls:
                stds.append(np.nanstd(l, ddof=0))
            return np.array(stds)

        def one_or_nan(x):
            y = np.ones(x.shape)
            y[np.isnan(x)] = float('nan')
            return y

        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        x_esr = ret['opinion_score_3darray']
        x_esr_original = ret['original_opinion_score_3darray']

        E, S, R = x_esr.shape
        C = dataset_reader.max_content_id_of_ref_videos + 1

        # === initialization ===

        mos = np.array(MosModel(dataset_reader).run_modeling()['quality_scores'])
        r_esr = x_esr - np.tile(mos, (S, 1)).T[:, :, None]  # r_esr: residual at e, s, r
        sigma_r_s = np.nanstd(SubjectiveModel._stack_repetitions_along_axis(r_esr, axis=0), axis=0, ddof=0)  # along e
        assert len(sigma_r_s) == S
        sigma_r_c = std_over_subject_and_content_id(r_esr, dataset_reader.content_id_of_dis_videos, C)
        assert len(sigma_r_c) == C

        x_e = mos # use MOS as initial value for x_e
        b_s = np.zeros(S)
        v_s = np.zeros(S) if cls.mode == 'SUBJECT_OBLIVIOUS' else sigma_r_s
        a_c = np.zeros(C) if cls.mode == 'CONTENT_OBLIVIOUS' else sigma_r_c

        x_e_std = None
        b_s_std = None
        v_s_std = None
        a_c_std = None

        # === iterations ===

        MAX_ITR = 10000
        REFRESH_RATE = 0.1
        EPSILON = 1e-3

        print('=== Belief Propagation ===')

        then = time.time()
        itr = 0
        while True:

            x_e_prev = x_e

            # ==== (12) b_s ====

            if gradient_method == 'simplified':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                num_num = x_esr - np.tile(x_e, (S, 1)).T[:, :, None]
                num_den = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                num = np.nansum(SubjectiveModel._stack_repetitions_along_axis(num_num / num_den, axis=0), axis=0)  # sum over e
                den_num = one_or_nan(x_esr) # 1 and nan
                den_den = num_den
                den = np.nansum(SubjectiveModel._stack_repetitions_along_axis(den_num / den_den, axis=0), axis=0)  # sum over e
                b_s_new = num / den
                b_s = b_s * (1.0 - REFRESH_RATE) + b_s_new * REFRESH_RATE
                b_s_std = 1.0 / np.sqrt(np.maximum(0., den))  # calculate std of x_e

            elif gradient_method == 'original':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                order1 = (x_esr - np.tile(x_e, (S, 1)).T[:, :, None] - np.tile(b_s, (E, 1))[:, :, None]) / vs2_add_ace2
                order1 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order1, axis=0), axis=0)  # sum over e
                order2 = - one_or_nan(x_esr) / vs2_add_ace2
                order2 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order2, axis=0), axis=0)  # sum over e
                b_s_new = b_s - order1 / order2
                b_s = b_s * (1.0 - REFRESH_RATE) + b_s_new * REFRESH_RATE
                b_s_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of x_e

            elif gradient_method == 'numerical':
                axis = 0  # sum over e
                order1 = (cls.loglikelihood_fcn(x_esr, x_e, b_s + EPSILON / 2.0, v_s, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf) -
                          cls.loglikelihood_fcn(x_esr, x_e, b_s - EPSILON / 2.0, v_s, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)) / EPSILON
                order2 = (cls.loglikelihood_fcn(x_esr, x_e, b_s + EPSILON, v_s, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)
                          - 2 * cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, dataset_reader.content_id_of_dis_videos,
                                                      axis, numerical_pdf)
                          + cls.loglikelihood_fcn(x_esr, x_e, b_s - EPSILON, v_s, a_c,
                                                  dataset_reader.content_id_of_dis_videos, axis,
                                                  numerical_pdf)) / EPSILON ** 2
                b_s_new = b_s - order1 / order2
                b_s = b_s * (1.0 - REFRESH_RATE) + b_s_new * REFRESH_RATE
                b_s_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of x_e

            else:
                assert False

            if cls.mode == 'SUBJECT_OBLIVIOUS':
                b_s = np.zeros(S)  # forcing zero, hence disabling
                b_s_std = np.zeros(S)

            # ==== (14) v_s ====

            if gradient_method == 'simplified':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                a_esr = x_esr - np.tile(x_e, (S, 1)).T[:, :, None] - np.tile(b_s, (E, 1))[:, :, None]
                vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                vs2_minus_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] - np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                num = - np.tile(v_s, (E, 1))[:, :, None] / vs2_add_ace2 + np.tile(v_s, (E, 1))[:, :, None] * a_esr**2 \
                      / vs2_add_ace2**2
                num = np.nansum(SubjectiveModel._stack_repetitions_along_axis(num, axis=0), axis=0)  # sum over e
                poly_term = np.tile(a_c_e**4, (S, 1)).T[:, :, None] \
                      - 3 * np.tile(v_s**4, (E, 1))[:, :, None] \
                      - 2 * np.tile(v_s**2, (E, 1))[:, :, None] * np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                den = vs2_minus_ace2 / vs2_add_ace2**2 + a_esr**2 * poly_term / vs2_add_ace2**4
                den = np.nansum(SubjectiveModel._stack_repetitions_along_axis(den, axis=0), axis=0)  # sum over e
                v_s_new = v_s - num / den
                v_s = v_s * (1.0 - REFRESH_RATE) + v_s_new * REFRESH_RATE
                # calculate std of v_s
                lpp = np.nansum(SubjectiveModel._stack_repetitions_along_axis(
                    vs2_minus_ace2 / vs2_add_ace2**2 + a_esr**2 * poly_term / vs2_add_ace2**4, axis=0),
                    axis=0)  # sum over e
                v_s_std = 1.0 / np.sqrt(np.maximum(0., -lpp))

            elif gradient_method == 'original':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                a_esr = x_esr - np.tile(x_e, (S, 1)).T[:, :, None] - np.tile(b_s, (E, 1))[:, :, None]
                vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                vs2_minus_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] - np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                poly_term = np.tile(a_c_e**4, (S, 1)).T[:, :, None] \
                      - 3 * np.tile(v_s**4, (E, 1))[:, :, None] \
                      - 2 * np.tile(v_s**2, (E, 1))[:, :, None] * np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                order1 = - np.tile(v_s, (E, 1))[:, :, None] / vs2_add_ace2 + \
                         np.tile(v_s, (E, 1))[:, :, None] * a_esr**2 / vs2_add_ace2 ** 2
                order1 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order1, axis=0), axis=0)  # sum over e
                order2 = vs2_minus_ace2 / vs2_add_ace2 ** 2 + a_esr ** 2 * poly_term / vs2_add_ace2 ** 4
                order2 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order2, axis=0), axis=0)  # sum over e
                v_s_new = v_s - order1 / order2
                v_s = v_s * (1.0 - REFRESH_RATE) + v_s_new * REFRESH_RATE
                v_s_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of v_s

            elif gradient_method == 'numerical':
                axis = 0  # sum over e
                order1 = (cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s + EPSILON / 2.0, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf) -
                          cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s - EPSILON / 2.0, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)) / EPSILON
                order2 = (cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s + EPSILON, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)
                          - 2 * cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, dataset_reader.content_id_of_dis_videos,
                                                      axis, numerical_pdf)
                          + cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s - EPSILON, a_c,
                                                  dataset_reader.content_id_of_dis_videos, axis,
                                                  numerical_pdf)) / EPSILON ** 2
                v_s_new = v_s - order1 / order2
                v_s = v_s * (1.0 - REFRESH_RATE) + v_s_new * REFRESH_RATE
                v_s_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of v_s

            else:
                assert False

            # force non-negative
            v_s = np.maximum(v_s, 0.0 * np.ones(v_s.shape))

            if cls.mode == 'SUBJECT_OBLIVIOUS':
                v_s = np.zeros(S) # forcing zero, hence disabling
                v_s_std = np.zeros(S)

            # ==== (15) a_c ====

            if gradient_method == 'simplified':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                a_esr = x_esr - np.tile(x_e, (S, 1)).T[:, :, None] - np.tile(b_s, (E, 1))[:, :, None]
                vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                vs2_minus_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] - np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                num = - np.tile(a_c_e, (S, 1)).T[:, :, None] / vs2_add_ace2 \
                      + np.tile(a_c_e, (S, 1)).T[:, :, None] * a_esr ** 2 / vs2_add_ace2 ** 2
                num = np.nansum(SubjectiveModel._stack_repetitions_along_axis(num, axis=1), axis=1)  # sum over s
                num = sum_over_content_id(num, dataset_reader.content_id_of_dis_videos, C)  # sum over e:c(e)=c
                poly_term = np.tile(v_s**4, (E, 1))[:, :, None] \
                      - 3 * np.tile(a_c_e**4, (S, 1)).T[:, :, None] \
                      - 2 * np.tile(v_s**2, (E, 1))[:, :, None] * np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                den = - vs2_minus_ace2 / vs2_add_ace2 ** 2 + a_esr ** 2 * poly_term / vs2_add_ace2 ** 4
                den = np.nansum(SubjectiveModel._stack_repetitions_along_axis(den, axis=1), axis=1)  # sum over s
                den = sum_over_content_id(den, dataset_reader.content_id_of_dis_videos, C)  # sum over e:c(e)=c
                # check: 'den' is 0 in test/subjective_model_test.py::SubjectiveModelPartialTest::test_observer_content_aware_subjective_model_nocontent
                a_c_new = a_c - num / den
                a_c = a_c * (1.0 - REFRESH_RATE) + a_c_new * REFRESH_RATE
                # calculate std of a_c
                lpp = sum_over_content_id(
                    np.nansum(
                        SubjectiveModel._stack_repetitions_along_axis(
                        -vs2_minus_ace2 / vs2_add_ace2 ** 2 + a_esr ** 2 * poly_term / vs2_add_ace2 ** 4,
                            axis=1),
                        axis=1),
                    dataset_reader.content_id_of_dis_videos,
                    C
                )  # sum over e:c(e)=c
                # check: max(0, ...) leads to division by zero
                a_c_std = 1.0 /np.sqrt(np.maximum(0., -lpp))

            elif gradient_method == 'original':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                a_esr = x_esr - np.tile(x_e, (S, 1)).T[:, :, None] - np.tile(b_s, (E, 1))[:, :, None]
                vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                vs2_minus_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] - np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                poly_term = np.tile(v_s**4, (E, 1))[:, :, None] \
                      - 3 * np.tile(a_c_e**4, (S, 1)).T[:, :, None] \
                      - 2 * np.tile(v_s**2, (E, 1))[:, :, None] * np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                order1 = - np.tile(a_c_e, (S, 1)).T[:, :, None] / vs2_add_ace2 \
                         + np.tile(a_c_e, (S, 1)).T[:, :, None] * a_esr ** 2 / vs2_add_ace2 ** 2
                order1 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order1, axis=1), axis=1)  # sum over s
                order1 = sum_over_content_id(order1, dataset_reader.content_id_of_dis_videos, C)  # sum over e:c(e)=c
                order2 = - vs2_minus_ace2 / vs2_add_ace2 ** 2 + a_esr ** 2 * poly_term / vs2_add_ace2 ** 4
                order2 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order2, axis=1), axis=1)  # sum over s
                order2 = sum_over_content_id(order2, dataset_reader.content_id_of_dis_videos, C)  # sum over e:c(e)=c
                a_c_new = a_c - order1 / order2
                a_c = a_c * (1.0 - REFRESH_RATE) + a_c_new * REFRESH_RATE
                a_c_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of a_c

            elif gradient_method == 'numerical':
                axis = 1  # sum over s
                order1 = (cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c + EPSILON / 2.0,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf) -
                          cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c - EPSILON / 2.0,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)) / EPSILON
                order2 = (cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c + EPSILON,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)
                          - 2 * cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, dataset_reader.content_id_of_dis_videos,
                                                      axis, numerical_pdf)
                          + cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c - EPSILON,
                                                  dataset_reader.content_id_of_dis_videos, axis,
                                                  numerical_pdf)) / EPSILON ** 2
                order1 = sum_over_content_id(order1, dataset_reader.content_id_of_dis_videos, C)  # sum over e:c(e)=c
                order2 = sum_over_content_id(order2, dataset_reader.content_id_of_dis_videos, C)  # sum over e:c(e)=c
                a_c_new = a_c - order1 / order2
                a_c = a_c * (1.0 - REFRESH_RATE) + a_c_new * REFRESH_RATE
                a_c_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of a_c

            else:
                assert False

            # force non-negative
            a_c = np.maximum(a_c, 0.0 * np.ones(a_c.shape))

            if cls.mode == 'CONTENT_OBLIVIOUS':
                a_c = np.zeros(C) # forcing zero, hence disabling
                a_c_std = np.zeros(C)

            # (11) ==== x_e ====

            if gradient_method == 'simplified':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                num_num = x_esr - np.tile(b_s, (E, 1))[:, :, None]
                num_den = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                num = np.nansum(SubjectiveModel._stack_repetitions_along_axis(num_num / num_den, axis=1), axis=1)  # sum over s
                den_num = one_or_nan(x_esr)  # 1 and nan
                den_den = num_den
                den = np.nansum(SubjectiveModel._stack_repetitions_along_axis(den_num / den_den, axis=1), axis=1)  # sum over s
                x_e_new = num / den
                x_e = x_e * (1.0 - REFRESH_RATE) + x_e_new * REFRESH_RATE
                x_e_std = 1.0 / np.sqrt(np.maximum(0., den))  # calculate std of x_e

            elif gradient_method == 'original':
                a_c_e = np.array([a_c[i] for i in dataset_reader.content_id_of_dis_videos])
                a_esr = x_esr - np.tile(x_e, (S, 1)).T[:, :, None] - np.tile(b_s, (E, 1))[:, :, None]
                vs2_add_ace2 = np.tile(v_s**2, (E, 1))[:, :, None] + np.tile(a_c_e**2, (S, 1)).T[:, :, None]
                order1 = a_esr / vs2_add_ace2
                order1 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order1, axis=1), axis=1)  # sum over s
                order2 = - one_or_nan(x_esr) / vs2_add_ace2
                order2 = np.nansum(SubjectiveModel._stack_repetitions_along_axis(order2, axis=1), axis=1)  # sum over s
                x_e_new = x_e - order1 / order2
                x_e = x_e * (1.0 - REFRESH_RATE) + x_e_new * REFRESH_RATE
                x_e_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of x_e

            elif gradient_method == 'numerical':
                axis = 1  # sum over s
                order1 = (cls.loglikelihood_fcn(x_esr, x_e + EPSILON / 2.0, b_s, v_s, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf) -
                          cls.loglikelihood_fcn(x_esr, x_e - EPSILON / 2.0, b_s, v_s, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)) / EPSILON
                order2 = (cls.loglikelihood_fcn(x_esr, x_e + EPSILON, b_s, v_s, a_c,
                                                dataset_reader.content_id_of_dis_videos, axis, numerical_pdf)
                          - 2 * cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, dataset_reader.content_id_of_dis_videos,
                                                      axis, numerical_pdf)
                          + cls.loglikelihood_fcn(x_esr, x_e - EPSILON, b_s, v_s, a_c,
                                                  dataset_reader.content_id_of_dis_videos, axis,
                                                  numerical_pdf)) / EPSILON ** 2
                x_e_new = x_e - order1 / order2
                x_e = x_e * (1.0 - REFRESH_RATE) + x_e_new * REFRESH_RATE
                x_e_std = 1.0 / np.sqrt(np.maximum(0., -order2))  # calculate std of x_e

            else:
                assert False

            itr += 1

            delta_x_e = linalg.norm(x_e_prev - x_e)

            loglikelihood = np.sum(
                cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, dataset_reader.content_id_of_dis_videos, 1,
                                      numerical_pdf))

            now = time.time()
            elapsed = now - then
            then = now

            msg = 'Iteration {itr:4d}: sec {sec:.1f}, change {delta_x_e}, loglikelihood {loglikelihood}, x_e {x_e}, b_s {b_s}, v_s {v_s}, a_c {a_c}'.\
                format(sec=elapsed, itr=itr, delta_x_e=delta_x_e, loglikelihood=loglikelihood, x_e=np.nanmean(x_e), b_s=np.nanmean(b_s), v_s=np.nanmean(v_s), a_c=np.nanmean(a_c))
            # msg = 'Iteration {itr:4d}: sec {sec:.1f}, change {delta_x_e}, likelihood {likelihood}, x_e {x_e}, b_s {b_s}, v_s {v_s}, a_c {a_c}'.\
            #     format(sec=elapsed, itr=itr, delta_x_e=delta_x_e, likelihood=likelihood, x_e=(np.min(x_e), np.mean(x_e), np.max(x_e)), b_s=(np.min(b_s), np.mean(b_s), np.max(b_s)), v_s=(np.min(v_s), np.mean(v_s), np.max(v_s)), a_c=(np.min(a_c), np.mean(a_c), np.max(a_c)))
            # msg = 'Iteration {itr:4d}: sec {sec:.1f}, change {delta_x_e}, likelihood {likelihood}, x_e {x_e}, b_s {b_s}, v_s (min) {v_s} ({v_s_min}), a_c (min) {a_c} ({a_c_min})'.\
            #     format(sec=elapsed, itr=itr, delta_x_e=delta_x_e, likelihood=likelihood, x_e=np.mean(x_e), b_s=np.mean(b_s), v_s=np.mean(v_s), v_s_min=np.min(v_s), a_c=np.mean(a_c), a_c_min=np.min(a_c))

            sys.stdout.write(msg + '\r')
            sys.stdout.flush()
            # print(msg)

            # time.sleep(0.001)

            if delta_x_e < delta_thr:
                break

            if itr >= MAX_ITR:
                break

        sys.stdout.write("\n")

        assert x_e_std is not None
        assert b_s_std is not None

        if force_subjbias_zeromean:
            mean_b_s = np.mean(b_s)
            b_s -= mean_b_s
            x_e += mean_b_s

        result = {
            'raw_scores': x_esr,

            'quality_scores': list(x_e),
            'quality_scores_std': list(x_e_std),
            'quality_scores_ci95': [list(1.95996 * x_e_std),
                                    list(1.95996 * x_e_std)],
            'num_iter': itr,
        }


        if cls.mode != 'SUBJECT_OBLIVIOUS':
            cnt_s = np.sum(~np.isnan(SubjectiveModel._stack_repetitions_along_axis(x_esr, axis=0)), axis=0)  # number of samples along i
            result['observer_bias'] = list(b_s)
            result['observer_bias_std'] = list(b_s_std)
            result['observer_bias_ci95'] = [list(1.95996 * b_s_std),
                                            list(1.95996 * b_s_std)]

            result['observer_inconsistency'] = list(v_s)
            result['observer_inconsistency_std'] = list(v_s_std)
            result['observer_inconsistency_ci95'] = [
                list((1 - np.sqrt(cnt_s / chi2.ppf(1 - 0.025, df=cnt_s))) * v_s),
                list((np.sqrt(cnt_s / chi2.ppf(0.025, df=cnt_s)) - 1) * v_s),
                # list(1.95996 * v_s_std),
                # list(1.95996 * v_s_std),
            ]

        if cls.mode != 'CONTENT_OBLIVIOUS':
            result['content_ambiguity'] = list(a_c)
            result['content_ambiguity_std'] = list(a_c_std)
            result['content_ambiguity_ci95'] = [list(1.95996 * a_c_std), list(1.95996 * a_c_std)]

        try:
            observers = dataset_reader._get_list_observers()  # may not exist
            result['observers'] = observers
        except AssertionError:
            pass

        result['reconstructions'] = cls._get_reconstructions(x_esr, x_e, b_s)

        original_E, original_S, original_R = x_esr_original.shape
        original_num_os = np.sum(~np.isnan(x_esr_original))
        original_C = dataset_reader.max_content_id_of_ref_videos + 1

        num_os = np.sum(~np.isnan(x_esr))

        dof = cls._get_dof(original_E, original_S, original_C, original_R) / original_num_os  # dof per observation
        result['dof'] = dof

        loglikelihood = np.sum(
            cls.loglikelihood_fcn(x_esr, x_e, b_s, v_s, a_c, dataset_reader.content_id_of_dis_videos, 1,
                                  numerical_pdf)) / num_os  # log-likelihood per observation
        result['loglikelihood'] = loglikelihood

        aic = 2 * dof - 2 * loglikelihood  # aic per observation
        result['aic'] = aic

        bic = np.log(original_num_os) * dof - 2 * loglikelihood  # bic per observation
        result['bic'] = bic

        return result

    @classmethod
    def _get_reconstructions(cls, x_esr, x_e, b_s):
        E, S, R = x_esr.shape
        x_esr_hat = np.zeros(x_esr.shape)
        for r in range(R):
            x_esr_hat[:, :, r] = np.tile(x_e, (S, 1)).T + np.tile(b_s, (E, 1))
        return x_esr_hat

    @classmethod
    def _get_dof(cls, E, S, C, R):
        if cls.mode == 'DEFAULT':
            dof = E + S * R * 2 + C
        elif cls.mode == 'CONTENT_OBLIVIOUS':
            dof = E + S * R * 2
        elif cls.mode == 'SUBJECT_OBLIVIOUS':
            dof = E + C
        else:
            assert False
        return dof


class MaximumLikelihoodEstimationModelContentOblivious(MaximumLikelihoodEstimationModel):
    TYPE = 'MLE_CO' # maximum likelihood estimation (no content modeling)
    VERSION = MaximumLikelihoodEstimationModel.VERSION + "_0.1"
    mode = 'CONTENT_OBLIVIOUS'


class MaximumLikelihoodEstimationModelSubjectOblivious(MaximumLikelihoodEstimationModel):
    TYPE = 'MLE_SO' # maximum likelihood estimation (no subject modeling)
    VERSION = MaximumLikelihoodEstimationModel.VERSION + "_0.1"
    mode = 'SUBJECT_OBLIVIOUS'


class SubjrejMosModel(MosModel):

    TYPE = 'SR_MOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['subject_rejection'] = True
        return super(SubjrejMosModel, self).run_modeling(**kwargs2)

class SubjrejMosModelPearson(MosModel):

    TYPE = 'SR_MOS_PCC'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['subject_rejection'] = True
        kwargs2['subject_rejection_type'] = 'pearson'
        return super(SubjrejMosModelPearson, self).run_modeling(**kwargs2)


class SubjrejMosModelSpearman(MosModel):

    TYPE = 'SR_MOS_SRCC'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['subject_rejection'] = True
        kwargs2['subject_rejection_type'] = 'spearman'
        return super(SubjrejMosModelSpearman, self).run_modeling(**kwargs2)


class ZscoringMosModel(MosModel):

    TYPE = 'ZS_MOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, '{} is already doing zscoring, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['zscore_mode'] = True
        return super(ZscoringMosModel, self).run_modeling(**kwargs2)


class ZscoringSubjrejMosModel(MosModel):

    TYPE = 'ZS_SR_MOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, '{} is already doing zscoring, no need to repeat.'.format(self.__class__.__name__)
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['zscore_mode'] = True
        kwargs2['subject_rejection'] = True
        return super(ZscoringSubjrejMosModel, self).run_modeling(**kwargs2)


class MaximumLikelihoodEstimationDmosModel(MaximumLikelihoodEstimationModel):

    TYPE = 'DMOS_MLE'
    VERSION = MaximumLikelihoodEstimationModel.VERSION + "_0.1"

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, '{} is already doing dscoring, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        return super(MaximumLikelihoodEstimationDmosModel, self).run_modeling(**kwargs2)


class MaximumLikelihoodEstimationDmosModelContentOblivious(MaximumLikelihoodEstimationModelContentOblivious):

    TYPE = 'DMOS_MLE_CO'
    VERSION = MaximumLikelihoodEstimationModelContentOblivious.VERSION + "_0.1"

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, '{} is already doing dscoring, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        return super(MaximumLikelihoodEstimationDmosModelContentOblivious, self).run_modeling(**kwargs2)


class SubjrejDmosModel(MosModel):

    TYPE = 'SR_DMOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, '{} is already doing dscoring, no need to repeat.'.format(self.__class__.__name__)
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        kwargs2['subject_rejection'] = True
        return super(SubjrejDmosModel, self).run_modeling(**kwargs2)


class ZscoringSubjrejDmosModel(MosModel):

    TYPE = 'ZS_SR_DMOS'
    VERSION = '0.1'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'dscore_mode' in kwargs and kwargs['dscore_mode'] is True:
            assert False, '{} is already doing dscoring, no need to repeat.'.format(self.__class__.__name__)
        if 'zscore_mode' in kwargs and kwargs['zscore_mode'] is True:
            assert False, '{} is already doing zscoring, no need to repeat.'.format(self.__class__.__name__)
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['dscore_mode'] = True
        kwargs2['zscore_mode'] = True
        kwargs2['subject_rejection'] = True
        return super(ZscoringSubjrejDmosModel, self).run_modeling(**kwargs2)


class PerSubjectModel(SubjectiveModel):
    """
    Subjective model that takes a raw dataset and output a 'per-subject dataset'
    with repeated disvideos, each assigned a per-subject score
    """
    TYPE = 'PERSUBJECT'
    VERSION = '1.0'

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        os_3darray = ret['opinion_score_3darray']

        result = {'quality_scores': os_3darray}
        return result

    def to_aggregated_dataset(self, **kwargs):
        self._assert_modeled()
        return self.dataset_reader.to_persubject_dataset(self.model_result['quality_scores'], **kwargs)

    def to_aggregated_dataset_file(self, dataset_filepath, **kwargs):
        self._assert_modeled()
        self.dataset_reader.to_persubject_dataset_file(dataset_filepath, self.model_result['quality_scores'], **kwargs)


class BiasremvMosModel(MosModel):

    TYPE = 'BR_MOS'
    VERSION = '1.0'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'bias_offset' in kwargs and kwargs['bias_offset'] is True:
            assert False, '{} is already doing bias offsetting, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['bias_offset'] = True
        return super(BiasremvMosModel, self).run_modeling(**kwargs2)

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):
        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        os_3darray = ret['opinion_score_3darray']
        original_os_3darray = ret['original_opinion_score_3darray']
        result = cls._get_mos_and_stats(os_3darray, original_os_3darray)
        result['observer_bias'] = list(ret['bias_offset_estimate'])
        if 'observer_rejected' in ret:
            result['observer_rejected'] = ret['observer_rejected']
            assert 'observer_rejected_1st_stats' in ret
            assert 'observer_rejected_2nd_stats' in ret
            result['observer_rejected_1st_stats'] = ret['observer_rejected_1st_stats']
            result['observer_rejected_2nd_stats'] = ret['observer_rejected_2nd_stats']
        return result

    @classmethod
    def _get_dof(cls, E, S, R):
        # override MosModel._get_dof
        return E * 2 + S * R


class BiasremvSubjrejMosModel(BiasremvMosModel):

    TYPE = 'BR_SR_MOS'
    VERSION = '1.0'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'bias_offset' in kwargs and kwargs['bias_offset'] is True:
            assert False, '{} is already doing bias offsetting, no need to repeat.'.format(self.__class__.__name__)
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['bias_offset'] = True
        kwargs2['subject_rejection'] = True
        result = super(BiasremvMosModel, self).run_modeling(**kwargs2)

        return result


class BiasremvSubjrejMosModelPearson(BiasremvMosModel):

    TYPE = 'BR_SR_MOS_PCC'
    VERSION = '1.0'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'bias_offset' in kwargs and kwargs['bias_offset'] is True:
            assert False, '{} is already doing bias offsetting, no need to repeat.'.format(self.__class__.__name__)
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['subject_rejection'] = True
        kwargs2['subject_rejection_type'] = 'pearson'
        result = super(BiasremvSubjrejMosModelPearson, self).run_modeling(**kwargs2)

        return result


class BiasremvSubjrejMosModelSpearman(BiasremvMosModel):

    TYPE = 'BR_SR_MOS_SRCC'
    VERSION = '1.0'

    def run_modeling(self, **kwargs):
        # override SubjectiveModel._run_modeling
        if 'bias_offset' in kwargs and kwargs['bias_offset'] is True:
            assert False, '{} is already doing bias offsetting, no need to repeat.'.format(self.__class__.__name__)
        if 'subject_rejection' in kwargs and kwargs['subject_rejection'] is True:
            assert False, '{} is already doing subject rejection, no need to repeat.'.format(self.__class__.__name__)
        kwargs2 = kwargs.copy()
        kwargs2['subject_rejection'] = True
        kwargs2['subject_rejection_type'] = 'spearman'
        result = super(BiasremvSubjrejMosModelSpearman, self).run_modeling(**kwargs2)

        return result


class SubjectMLEModelProjectionSolver(SubjectiveModel):

    TYPE = 'Subject_MLE_Projection'
    VERSION = '0.1'

    @staticmethod
    def _one_or_nan(x):
        y = np.ones(x.shape)
        y[np.isnan(x)] = float('nan')
        return y

    @classmethod
    def _run_modeling(cls, dataset_reader, **kwargs):

        force_subjbias_zeromean = kwargs['force_subjbias_zeromean'] if \
            'force_subjbias_zeromean' in kwargs and kwargs['force_subjbias_zeromean'] is not None else True
        assert isinstance(force_subjbias_zeromean, bool)

        ret = cls._get_opinion_score_3darray_with_preprocessing(dataset_reader, **kwargs)
        x_jir = ret['opinion_score_3darray']
        x_jir_original = ret['original_opinion_score_3darray']
        J, I, R = x_jir.shape
        cnt_i = np.sum(~np.isnan(SubjectiveModel._stack_repetitions_along_axis(x_jir, axis=0)), axis=0)  # number of samples along i
        cnt_j = np.sum(~np.isnan(SubjectiveModel._stack_repetitions_along_axis(x_jir, axis=1)), axis=1)  # number of samples along j

        # video by video, estimate MOS by averaging over subjects
        s_j = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(x_jir, axis=1), axis=1)  # mean marginalized over i

        # subject by subject, estimate subject bias by comparing with MOS
        b_jir = x_jir - np.tile(s_j, (I, 1)).T[:, :, None]
        b_i = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(b_jir, axis=0), axis=0)  # mean marginalized over j

        MAX_ITR = 1000
        DELTA_THR = 1e-8
        EPSILON = 1e-8

        itr = 0
        while True:

            s_j_prev = s_j

            # subject by subject, estimate subject inconsistency by averaging the residue over stimuli
            r_jir = x_jir - np.tile(s_j, (I, 1)).T[:, :, None] - np.tile(b_i, (J, 1))[:, :, None]
            v_i = np.nanstd(SubjectiveModel._stack_repetitions_along_axis(r_jir, axis=0), axis=0)
            v_j = np.nanstd(SubjectiveModel._stack_repetitions_along_axis(r_jir, axis=1), axis=1)

            # video by video, estimate MOS by averaging over subjects, inversely weighted by residue variance
            s_jir = x_jir - np.tile(b_i, (J, 1))[:, :, None]
            w_i = 1.0 / (v_i ** 2 + EPSILON)
            s_j = weighed_nanmean_2d(
                SubjectiveModel._stack_repetitions_along_axis(s_jir, axis=1),
                weights=np.tile(w_i, R),
                axis=1)  # mean marginalized over i

            # subject by subject, estimate subject bias by comparing with MOS, inversely weighted by residue variance
            b_jir = x_jir - np.tile(s_j, (I, 1)).T[:, :, None]
            b_i = np.nanmean(SubjectiveModel._stack_repetitions_along_axis(b_jir, axis=0), axis=0)  # mean marginalized over j

            itr += 1

            delta_s_j = linalg.norm(s_j_prev - s_j)

            msg = 'Iteration {itr:4d}: change {delta_s_j}, s_j {s_j}, b_i {b_i}, v_i {v_i}'.format(
                itr=itr, delta_s_j=delta_s_j, s_j=np.mean(s_j), b_i=np.mean(b_i), v_i=np.mean(v_i))

            sys.stdout.write(msg + '\r')
            sys.stdout.flush()

            if delta_s_j < DELTA_THR:
                break

            if itr >= MAX_ITR:
                break
        s_j_std = cls._get_s_j_std(v_i, v_j, x_jir)

        den = np.nansum(
            SubjectiveModel._stack_repetitions_along_axis(
                cls._one_or_nan(x_jir) / np.tile(v_i ** 2, (x_jir.shape[0], 1))[:, :, None],
                axis=0),
            axis=0)  # sum over e
        b_i_std = 1.0 / np.sqrt(np.maximum(0., den))  # calculate std of b_i

        r_jir = x_jir - np.tile(s_j, (I, 1)).T[:, :, None] - np.tile(b_i, (J, 1))[:, :, None]
        v_i2 = np.tile(v_i ** 2, (J, 1))[:, :, None]
        poly_term = - 3 * np.tile(v_i ** 4, (J, 1))[:, :, None]
        lpp = np.nansum(
            SubjectiveModel._stack_repetitions_along_axis(
                1.0 / v_i2 + r_jir ** 2 * poly_term / v_i2 ** 4,
                axis=0),
            axis=0)  # sum over e
        v_i_std = 1.0 / np.sqrt(np.maximum(0., -lpp))

        sys.stdout.write("\n")

        if force_subjbias_zeromean:
            mean_b_i = np.mean(b_i)
            b_i -= mean_b_i
            s_j += mean_b_i

        x_ji = SubjectiveModel._stack_repetitions_along_axis(x_jir, axis=0)

        result = {'raw_scores': x_jir,
                  'quality_scores': list(s_j),
                  'quality_scores_std': list(s_j_std),
                  'quality_scores_ci95': [list(1.95996 * s_j_std),
                                          list(1.95996 * s_j_std)],
                  'observer_bias': list(b_i), 'observer_bias_std': list(b_i_std),
                  'observer_bias_ci95': [list(1.95996 * b_i_std),
                                         list(1.95996 * b_i_std)],
                  'observer_inconsistency': list(v_i),
                  'observer_inconsistency_std': list(v_i_std),
                  'observer_inconsistency_ci95': [
                      list((1 - np.sqrt(cnt_i / chi2.ppf(1-0.025, df=cnt_i))) * v_i),
                      list((np.sqrt(cnt_i / chi2.ppf(0.025, df=cnt_i)) - 1) * v_i),
                      # list(1.95996 * v_i_std),
                      # list(1.95996 * v_i_std),
                  ],
                  'observer_scores_mean': list(np.nanmean(x_ji, axis=0)),
                  'observer_scores_std': list(np.nanstd(x_ji, axis=0)),
                  'reconstructions': cls._get_reconstructions(x_jir, s_j, b_i),
                  'num_iter': itr,
                  }

        original_J, original_I, original_R = x_jir_original.shape
        original_num_os = np.sum(~np.isnan(x_jir_original))

        num_os = np.sum(~np.isnan(x_jir))

        dof = (original_J + original_I * original_R * 2) / original_num_os
        result['dof'] = dof

        loglikelihood = cls.loglikelihood_function(np.hstack([s_j, b_i, v_i]), x_jir) / num_os
        result['loglikelihood'] = loglikelihood

        aic = 2 * dof - 2 * loglikelihood  # aic per observation
        result['aic'] = aic

        bic = np.log(original_num_os) * dof - 2 * loglikelihood  # bic per observation
        result['bic'] = bic

        return result

    @classmethod
    def _get_s_j_std(cls, v_i, v_j, x_jir):
        den = np.nansum(
            SubjectiveModel._stack_repetitions_along_axis(
                cls._one_or_nan(x_jir) / np.tile(v_i ** 2, (x_jir.shape[0], 1))[:, :, None],
                axis=1),
            axis=1)  # sum over s
        s_j_std = 1.0 / np.sqrt(np.maximum(0., den))  # calculate std of s_j
        return s_j_std

    @classmethod
    def _get_reconstructions(cls, x_jir, s_j, b_i):
        J, I, R = x_jir.shape
        x_jir_hat = np.zeros(x_jir.shape)
        for r in range(R):
            x_jir_hat[:, :, r] = np.tile(s_j, (I, 1)).T + np.tile(b_i, (J, 1))
        return x_jir_hat

    @staticmethod
    def loglikelihood_function(x, x_jir):
        J, I, R = x_jir.shape
        assert len(x) == J + I + I
        x_j, b_i, v_i = x[0: J], x[J: J + I], x[J + I: J + 2 * I]

        mtx = np.log(norm.pdf(
            x_jir,
            loc=np.tile(x_j, (I, 1)).T[:, :, None] + np.tile(b_i, (J, 1))[:, :, None],
            scale=np.tile(v_i, (J, 1))[:, :, None]
        ))
        # --- can be simplified as: ---
        # a_ji = x_ji - np.tile(x_j, (I, 1)).T - np.tile(b_i, (J, 1))
        # mtx = - 0.5 * np.tile(np.log(v_i**2), (J, 1)) - 0.5 * (a_ji) ** 2 / np.tile(v_i**2, (J, 1))

        ll = mtx[~np.isnan(x_jir)].sum()
        return ll


class SubjectMLEModelProjectionSolverAltSubjStdMixin(object):
    @classmethod
    def _get_s_j_std(cls, v_i, v_j, x_jir):
        den = np.nansum(
            SubjectiveModel._stack_repetitions_along_axis(
                cls._one_or_nan(x_jir) / np.tile(v_j ** 2, (x_jir.shape[1], 1)).T[:, :, None],
                axis=1),
            axis=1)  # sum over s
        s_j_std = 1.0 / np.sqrt(np.maximum(0., den))  # calculate std of s_j
        return s_j_std


class SubjectMLEModelProjectionSolver2(SubjectMLEModelProjectionSolverAltSubjStdMixin, SubjectMLEModelProjectionSolver):

    TYPE = 'Subject_MLE_Projection2'


class MaximumLikelihoodEstimationModelContentObliviousAlternativeProjection(SubjectMLEModelProjectionSolver):
    TYPE = 'MLE_CO_AP'


class MaximumLikelihoodEstimationModelContentObliviousAlternativeProjection2(SubjectMLEModelProjectionSolver2):
    TYPE = 'MLE_CO_AP2'
