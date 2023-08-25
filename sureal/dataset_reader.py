import pprint
import copy
import random

import numpy as np

from sureal.tools.misc import empty_object, get_unique_sorted_list
from sureal.tools.decorator import memoized as persist, override
from sureal.tools.misc import get_unique_sorted_list

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class DatasetReader(object):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self._assert_dataset()

    def _assert_dataset(self):
        # assert content id is from 0 to the total_content - 1
        cids = []
        for ref_video in self.dataset.ref_videos:
            cids.append(ref_video['content_id'])
        expected_cids = range(np.max(cids) + 1)
        for cid in cids:
            assert cid in expected_cids, \
                'reference video content_ids must be in [0, {}), but is {}'.\
                    format(self.num_ref_videos, cid)

        # assert dis_video content_id is content_ids
        for dis_video in self.dis_videos:
            assert dis_video['content_id'] in cids, \
                "dis_video of content_id {content_id}, asset_id {asset_id} must have content_id in {cids}".format(
                    content_id=dis_video['content_id'], asset_id=dis_video['asset_id'], cids=cids)

    @property
    def dis_videos(self):
        return self.dataset.dis_videos

    @property
    def num_dis_videos(self):
        return len(self.dis_videos)

    @property
    def num_ref_videos(self):
        return len(self.dataset.ref_videos)

    @property
    def max_content_id_of_ref_videos(self):
        return max(map(lambda ref_video: ref_video['content_id'], self.dataset.ref_videos))

    @property
    def content_ids(self):
        return list(set(map(lambda ref_video: ref_video['content_id'], self.dataset.ref_videos)))

    @property
    def asset_ids(self):
        return list(set(map(lambda dis_video: dis_video['asset_id'], self.dis_videos)))

    @property
    def content_id_of_dis_videos(self):
        return list(map(lambda dis_video: dis_video['content_id'], self.dis_videos))

    @property
    def _contentid_to_refvideo_map(self):
        d = {}
        for ref_video in self.dataset.ref_videos:
            d[ref_video['content_id']] = ref_video
        return d

    @property
    def disvideo_is_refvideo(self):
        d = self._contentid_to_refvideo_map
        return list(map(
            lambda dis_video: d[dis_video['content_id']]['path'] == dis_video['path'],
            self.dis_videos
        ))

    @property
    def ref_score(self):
        return self.dataset.ref_score if hasattr(self.dataset, 'ref_score') else None

    def to_dataset(self):
        return self.dataset

    @staticmethod
    def write_out_dataset(dataset, output_dataset_filepath):
        assert (hasattr(dataset, 'ref_videos'))
        assert (hasattr(dataset, 'dis_videos'))
        # write out
        with open(output_dataset_filepath, 'wt') as output_file:
            for key in dataset.__dict__.keys():
                if key != 'ref_videos' and key != 'dis_videos' \
                        and key != 'subjects' and not key.startswith('__'):
                    output_file.write('{} = '.format(key) + repr(
                        dataset.__dict__[key]) + '\n')
            output_file.write('\n')
            output_file.write('ref_videos = ' + pprint.pformat(
                dataset.ref_videos) + '\n')
            output_file.write('\n')
            output_file.write('dis_videos = ' + pprint.pformat(
                dataset.dis_videos) + '\n')
            if 'subjects' in dataset.__dict__.keys():
                output_file.write('\n')
                output_file.write('subjects = ' + pprint.pformat(
                    dataset.subjects) + '\n')
            if 'license' in dataset.__dict__.keys():
                output_file.write('\n')
                output_file.write('license = ' + pprint.pformat(
                    dataset.license) + '\n')


class RawDatasetReader(DatasetReader):
    """
    Reader for a subjective quality test dataset with raw scores (dis_video must
    have key of 'os' (opinion score)).
    """

    def _assert_dataset(self):
        """
        Override DatasetReader._assert_dataset
        """

        super(RawDatasetReader, self)._assert_dataset()

        # assert each dis_video dict has key 'os' (opinion score), and must
        # be iterable (list, tuple or dictionary)
        for dis_video in self.dis_videos:
            assert 'os' in dis_video, "dis_video must have key 'os' (opinion score)"
            assert isinstance(dis_video['os'], (list, tuple, dict))

        # make sure each dis video has equal number of observers
        if (
                    isinstance(self.dis_videos[0]['os'], list) or
                    isinstance(self.dis_videos[0]['os'], tuple)
        ):
            num_observers = len(self.dis_videos[0]['os'])
            for dis_video in self.dis_videos[1:]:
                assert num_observers == len(dis_video['os']), \
                    "expect number of observers {expected} but got {actual} for {dis_video}".format(
                        expected=num_observers, actual=len(dis_video['os']), dis_video=str(dis_video))

    def _get_num_observers(self):
        if (
                    isinstance(self.dis_videos[0]['os'], list) or
                    isinstance(self.dis_videos[0]['os'], tuple)
        ):
            return len(self.dis_videos[0]['os'])
        elif isinstance(self.dis_videos[0]['os'], dict):
            list_observers = self._get_list_observers()
            return len(list_observers)
        else:
            assert False, ''

    @property
    def num_observers(self):
        return self._get_num_observers()

    def _get_list_observers(self):

        for dis_video in self.dis_videos:
            assert isinstance(dis_video['os'], dict)

        list_observers = []
        for dis_video in self.dis_videos:
            list_observers += dis_video['os'].keys()

        return get_unique_sorted_list(list_observers)

    def _get_max_repetitions(self):
        """get the maximum number of times any observer evaluated any stimulus. For example, if all of the observers
        evaluated each stimulus only once, the function returns 1.
        """
        max_reps = 1
        if (
                    isinstance(self.dis_videos[0]['os'], list) or
                    isinstance(self.dis_videos[0]['os'], tuple)
        ):
            for dis_video in self.dis_videos:
                for idx_obs in range(self.num_observers):
                    scores = dis_video['os'][idx_obs]
                    if isinstance(scores, list) or isinstance(scores, tuple):
                        if len(scores) > max_reps:
                            max_reps = len(scores)

        elif isinstance(self.dis_videos[0]['os'], dict):
            for dis_video in self.dis_videos:
                list_observers = dis_video['os'].keys()

                for observer in list_observers:
                    scores = dis_video['os'][observer]
                    if isinstance(scores, list) or isinstance(scores, tuple):
                        if len(scores) > max_reps:
                            max_reps = len(scores)
        else:
            assert False, ''

        return max_reps

    @property
    def max_repetitions(self):
        return self._get_max_repetitions()

    @property
    def opinion_score_3darray(self):
        """
        3darray storing raw opinion scores, with first dimension the number of
        distorted videos, second dimension the number of observers, and third the maximum number of repetitions
        """
        score_mtx = float('NaN') * np.ones([self.num_dis_videos, self._get_num_observers(), self._get_max_repetitions()])

        if isinstance(self.dis_videos[0]['os'], list) \
                or isinstance(self.dis_videos[0]['os'], tuple):
            for i_dis_video, dis_video in enumerate(self.dis_videos):
                for i_observer in range(self.num_observers):
                    if isinstance(dis_video['os'][i_observer], list) or isinstance(dis_video['os'][i_observer], tuple):
                        reps = len(dis_video['os'][i_observer])
                    else:
                        reps = 1
                    score_mtx[i_dis_video, i_observer, :reps] = dis_video['os'][i_observer]
        elif isinstance(self.dis_videos[0]['os'], dict):
            list_observers = self._get_list_observers()
            for i_dis_video, dis_video in enumerate(self.dis_videos):
                for i_observer, observer in enumerate(list_observers):
                    if observer in dis_video['os']:
                        if isinstance(dis_video['os'][observer], list) or isinstance(dis_video['os'][observer], tuple):
                            reps = len(dis_video['os'][observer])
                        else:
                            reps = 1
                        score_mtx[i_dis_video, i_observer, :reps] = dis_video['os'][observer]
        else:
            assert False
        return score_mtx

    def to_aggregated_dataset(self, aggregate_scores, **kwargs):

        newone = self._prepare_new_dataset(kwargs)

        # ref_videos: deepcopy
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)

        # dis_videos: use input aggregate scores
        dis_videos = []
        assert len(self.dis_videos) == len(aggregate_scores)
        for dis_video, score in zip(self.dis_videos, aggregate_scores):
            dis_video2 = copy.deepcopy(dis_video)
            if 'os' in dis_video2: # remove 'os' - opinion score
                del dis_video2['os']
            if isinstance(score, np.ndarray):
                if len(score) == 1:
                    dis_video2['groundtruth'] = float(score)
                else:
                    dis_video2['groundtruth'] = list(score)
            else:
                dis_video2['groundtruth'] = score
            dis_videos.append(dis_video2)

        # add scores std if available
        if 'scores_std' in kwargs and kwargs['scores_std'] is not None:
            assert len(dis_videos) == len(kwargs['scores_std'])
            for dis_video, score_std in zip(dis_videos, kwargs['scores_std']):
                dis_video['groundtruth_std'] = score_std

        if 'aggregate_content_ids' in kwargs and kwargs['aggregate_content_ids'] is not None:
            dis_videos = list(filter(lambda dis_video: dis_video['content_id'] in kwargs['aggregate_content_ids'], dis_videos))

        if 'aggregate_asset_ids' in kwargs and kwargs['aggregate_asset_ids'] is not None:
            dis_videos = list(filter(lambda dis_video: dis_video['asset_id'] in kwargs['aggregate_asset_ids'], dis_videos))

        newone.dis_videos = dis_videos

        return newone

    def _prepare_new_dataset(self, kwargs):
        newone = empty_object()
        # systematically copy fields, e.g. dataset_name, yuv_fmt, width, height, ...
        for key in self.dataset.__dict__.keys():
            if not key.startswith('__'):  # filter out those e.g. __builtin__ ...
                setattr(newone, key, getattr(self.dataset, key))
        if 'quality_width' in kwargs and kwargs['quality_width'] is not None:
            newone.quality_width = kwargs['quality_width']
        elif hasattr(self.dataset, 'quality_width'):
            newone.quality_width = self.dataset.quality_width
        if 'quality_height' in kwargs and kwargs['quality_height'] is not None:
            newone.quality_height = kwargs['quality_height']
        elif hasattr(self.dataset, 'quality_height'):
            newone.quality_height = self.dataset.quality_height
        if 'resampling_type' in kwargs and kwargs['resampling_type'] is not None:
            newone.resampling_type = kwargs['resampling_type']
        elif hasattr(self.dataset, 'resampling_type'):
            newone.resampling_type = self.dataset.resampling_type
        return newone

    def to_aggregated_dataset_file(self, dataset_filepath, aggregate_scores, **kwargs):
        aggregate_dataset = self.to_aggregated_dataset(aggregate_scores, **kwargs)
        self.write_out_dataset(aggregate_dataset, dataset_filepath)

    def to_persubject_dataset(self, quality_scores, **kwargs):

        import math

        newone = self._prepare_new_dataset(kwargs)

        # ref_videos: deepcopy
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)

        # dis_videos: use input aggregate scores
        dis_videos = []
        for dis_video, quality_score in zip(self.dis_videos, quality_scores):
            assert 'os' in dis_video

            # quality_score should be a 1-D array with (processed) per-subject scores
            assert hasattr(quality_score, '__len__')

            # new style: opinion is specified as a dict: user -> score. In this
            # case, quality_score may contain nan. In this case: filter them out
            if isinstance(dis_video['os'], dict):
                quality_score = list(filter(lambda x: not math.isnan(x), quality_score))

            assert len(dis_video['os']) == len(quality_score)

            for persubject_score in quality_score:
                dis_video2 = copy.deepcopy(dis_video)
                if 'os' in dis_video2: # remove 'os' - opinion score
                    del dis_video2['os']
                if isinstance(persubject_score, np.ndarray):
                    if len(persubject_score) == 1:
                        dis_video2['groundtruth'] = float(persubject_score)
                    else:
                        dis_video2['groundtruth'] = list(persubject_score)
                else:
                    dis_video2['groundtruth'] = persubject_score
                dis_videos.append(dis_video2)
        newone.dis_videos = dis_videos

        return newone

    def to_persubject_dataset_file(self, dataset_filepath, quality_scores, **kwargs):
        persubject_dataset = self.to_persubject_dataset(quality_scores, **kwargs)
        self.write_out_dataset(persubject_dataset, dataset_filepath)

    def to_pc_dataset(self, **kwargs):

        newone = self._prepare_new_dataset(kwargs)

        # ref_videos: deepcopy
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)

        pc_type = kwargs['pc_type'] if 'pc_type' in kwargs and kwargs['pc_type'] is not None else 'within_subject_within_content'
        tiebreak_method = kwargs['tiebreak_method'] if 'tiebreak_method' in kwargs and kwargs['tiebreak_method'] is not None else 'even_split'

        sampling_seed = kwargs['sampling_seed'] if 'sampling_seed' in kwargs and kwargs['sampling_seed'] is not None else None

        sampling_rate = kwargs['sampling_rate'] if 'sampling_rate' in kwargs and kwargs['sampling_rate'] is not None else None
        per_asset_sampling_rates = kwargs['per_asset_sampling_rates'] if 'per_asset_sampling_rates' in kwargs and kwargs['per_asset_sampling_rates'] is not None else None

        cointoss_rate = kwargs['cointoss_rate'] if 'cointoss_rate' in kwargs and kwargs['cointoss_rate'] is not None else None
        per_asset_cointoss_rates = kwargs['per_asset_cointoss_rates'] if 'per_asset_cointoss_rates' in kwargs and kwargs['per_asset_cointoss_rates'] is not None else None

        noise_level = kwargs['noise_level'] if 'noise_level' in kwargs and kwargs['noise_level'] is not None else None
        per_asset_noise_levels = kwargs['per_asset_noise_levels'] if 'per_asset_noise_levels' in kwargs and kwargs['per_asset_noise_levels'] is not None else None

        per_asset_mean_scores = kwargs['per_asset_mean_scores'] if 'per_asset_mean_scores' in kwargs and kwargs['per_asset_mean_scores'] is not None else None

        assert pc_type == 'within_subject_within_content' or pc_type == 'within_subject'
        assert tiebreak_method == 'even_split' or tiebreak_method == 'coin_toss'

        assert not (sampling_rate is not None and per_asset_sampling_rates is not None)
        if sampling_rate is not None:
            assert np.isscalar(sampling_rate) and 0.0 <= sampling_rate
        if per_asset_sampling_rates is not None:
            assert len(per_asset_sampling_rates) == len(self.dis_videos)
            for per_asset_sampling_rate in per_asset_sampling_rates:
                assert np.isscalar(per_asset_sampling_rate) and 0.0 <= per_asset_sampling_rate

        assert not (cointoss_rate is not None and per_asset_cointoss_rates is not None)
        if cointoss_rate is not None:
            assert np.isscalar(cointoss_rate) and 0.0 <= cointoss_rate <= 1.0
        if per_asset_cointoss_rates is not None:
            assert len(per_asset_cointoss_rates) == len(self.dis_videos)
            for cointoss_rate_ in per_asset_cointoss_rates:
                assert np.isscalar(cointoss_rate_) and 0.0 <= cointoss_rate_ <= 1.0

        assert not (noise_level is not None and per_asset_noise_levels is not None)
        if noise_level is not None:
            assert np.isscalar(noise_level) and 0.0 <= noise_level
        if per_asset_noise_levels is not None:
            assert len(per_asset_noise_levels) == len(self.dis_videos)
            for noise_level_ in per_asset_noise_levels:
                assert np.isscalar(noise_level_) and 0.0 <= noise_level_

        if per_asset_mean_scores is not None:
            assert len(per_asset_mean_scores) == len(self.dis_videos)
            for mean_score_ in per_asset_mean_scores:
                assert np.isscalar(mean_score_)

        dis_videos = self.dis_videos
        if isinstance(dis_videos[0]['os'], dict):
            pass
        elif isinstance(dis_videos[0]['os'], (list, tuple)):
            # converting to dict_style
            for dis_video in dis_videos:
                scores = dis_video['os']
                dis_video['os'] = dict(zip(map(lambda x: str(x), range(len(scores))), scores))
        else:
            assert False

        # build nested subject-asset_id dict: subj -> (asset_id -> {'score': score, 'content_id': content_id, ...})
        d_subj_assetid = dict()
        for dis_video in dis_videos:
            for subj in dis_video['os']:
                if subj not in d_subj_assetid:
                    d_subj_assetid[subj] = dict()
                assert dis_video['asset_id'] not in d_subj_assetid[subj] # assuming no repetition for single subject and a dis_video
                d_subj_assetid[subj][dis_video['asset_id']] = {'score': dis_video['os'][subj], 'content_id': dis_video['content_id']}

        # prepare new dis_videos, and create index from asset_id to dis_videos
        new_dis_videos = copy.deepcopy(dis_videos)
        d_assetid_disvideoidx = dict() # build dict: asset_id -> index of dis_videos
        for i_dis_video, dis_video in enumerate(new_dis_videos):
            dis_video['os'] = dict()
            d_assetid_disvideoidx[dis_video['asset_id']] = i_dis_video

        # set seed
        if sampling_seed is not None:
            random.seed(sampling_seed)

        # iterate through subj, compare asset_id pairs (upper triangle only), put pc results into new
        for subj in d_subj_assetid:
            assetids = sorted(d_subj_assetid[subj].keys())
            for idx in range(len(assetids)):
                for idx2 in range(idx):
                    assetid = assetids[idx]
                    assetid2 = assetids[idx2]
                    content_id = d_subj_assetid[subj][assetid]['content_id']
                    content_id2 = d_subj_assetid[subj][assetid2]['content_id']

                    if pc_type == 'within_subject_within_content':
                        if content_id != content_id2:
                            continue
                    elif pc_type == 'within_subject':
                        pass
                    else:
                        assert False, "unknown pc_type: {}".format(pc_type)

                    if sampling_rate is not None or per_asset_sampling_rates is not None:
                        if sampling_rate is not None:
                            true_sampling_rate = sampling_rate
                        elif per_asset_sampling_rates is not None:
                            # the true sampling rate of a pair is the mean of the sampling rate of the two assets:
                            true_sampling_rate = (per_asset_sampling_rates[idx] + per_asset_sampling_rates[idx2]) / 2.0
                        else:
                            assert False
                    else:
                        true_sampling_rate = 1.0

                    while true_sampling_rate > 0.0:

                        if true_sampling_rate >= 1.0:
                            true_sampling_rate -= 1.0
                            pass
                        elif true_sampling_rate > 0.0:  # true_sampling_rate is within (0.0, 1.0)
                            if random.random() > true_sampling_rate:
                                true_sampling_rate = 0.0
                                continue
                            true_sampling_rate = 0.0
                        else:  # true_sampling_rate is 0.0
                            break

                        if cointoss_rate is not None:
                            true_cointoss_rate = cointoss_rate
                        elif per_asset_cointoss_rates is not None:
                            true_cointoss_rate = (per_asset_cointoss_rates[idx] + per_asset_cointoss_rates[idx2]) / 2.0
                        else:
                            true_cointoss_rate = None

                        if true_cointoss_rate is not None and random.random() < true_cointoss_rate:
                            if random.random() > 0.5:
                                new_dis_videos[d_assetid_disvideoidx[assetid]]['os'][(subj, assetid2)] = 1
                            else:
                                new_dis_videos[d_assetid_disvideoidx[assetid2]]['os'][(subj, assetid)] = 1
                        else:

                            if per_asset_mean_scores is not None:
                                mscore = per_asset_mean_scores[assetid]
                                mscore2 = per_asset_mean_scores[assetid2]
                            else:
                                mscore = d_subj_assetid[subj][assetid]['score']
                                mscore2 = d_subj_assetid[subj][assetid2]['score']

                            if noise_level is not None:
                                score = mscore + random.gauss(0, noise_level)
                                score2 = mscore2 + random.gauss(0, noise_level)
                            elif per_asset_noise_levels is not None:
                                score = mscore + random.gauss(0, per_asset_noise_levels[idx])
                                score2 = mscore2 + random.gauss(0, per_asset_noise_levels[idx2])
                            else:
                                score = mscore
                                score2 = mscore2

                            if score > score2:
                                new_dis_videos[d_assetid_disvideoidx[assetid]]['os'][(subj, assetid2)] = 1
                            elif score < score2:
                                new_dis_videos[d_assetid_disvideoidx[assetid2]]['os'][(subj, assetid)] = 1
                            else:
                                if tiebreak_method == 'even_split':
                                    # each one gets fair share
                                    new_dis_videos[d_assetid_disvideoidx[assetid]]['os'][(subj, assetid2)] = 0.5
                                    new_dis_videos[d_assetid_disvideoidx[assetid2]]['os'][(subj, assetid)] = 0.5
                                elif tiebreak_method == 'coin_toss':
                                    if random.random() > 0.5:
                                        new_dis_videos[d_assetid_disvideoidx[assetid]]['os'][(subj, assetid2)] = 1
                                    else:
                                        new_dis_videos[d_assetid_disvideoidx[assetid2]]['os'][(subj, assetid)] = 1
                                else:
                                    assert False, "unknown tiebreak_method: {}".format(tiebreak_method)

        newone.dis_videos = new_dis_videos

        return newone

    def to_pc_dataset_file(self, dataset_filepath, **kwargs):
        pc_dataset = self.to_pc_dataset(**kwargs)
        self.write_out_dataset(pc_dataset, dataset_filepath)

    def to_combined_overlap_dataset(self, second_dataset_reader, **kwargs):
        """
        A function to find an overlap between self.dataset and the second dataset and combine the scores for the
        overlapping videos. The overlap is determined by matching paths.
        """
        assert isinstance(second_dataset_reader, RawDatasetReader), 'RawDatasetReader can only be combined with ' \
                                                                    'another RawDatasetReader'

        # make both of the datasets dictionary style
        first_dataset = self.to_dictionary_style_dataset()
        second_dataset = second_dataset_reader.to_dictionary_style_dataset()

        newone = self._prepare_new_dataset(kwargs)
        newone.ref_videos = []
        newone.dis_videos = []

        content_ids_in = []
        for vid1 in first_dataset.dis_videos:
            for vid2 in second_dataset.dis_videos:
                if vid1['path'] == vid2['path']:  # TODO: Perhaps worth modifying so that the path does not need to match exactly

                    # if the reference video isn't already in the newone.ref_videos then add it
                    if vid1['content_id'] not in content_ids_in:
                        content_ids_in.append(vid1['content_id'])
                        new_content_id = content_ids_in.index(vid1['content_id'])
                        for ref in first_dataset.ref_videos:
                            if ref['content_id'] == vid1['content_id']:
                                new_ref = copy.deepcopy(ref)
                                new_ref['content_id'] = new_content_id
                                newone.ref_videos.append(new_ref)
                                break
                    else:
                        new_content_id = content_ids_in.index(vid1['content_id'])

                    new_dis_video = copy.deepcopy(vid1)
                    new_dis_video['content_id'] = new_content_id

                    new_dis_video['os'].update(vid2['os'])  # combined the 'os's but overrode when subject is in both

                    for key in vid2['os'].keys():
                        if key in vid1['os'].keys():  # the same subject is in both datasets
                            if not isinstance(vid1['os'][key], list):
                                new_os = [vid1['os'][key]]
                            else:
                                new_os = vid1['os'][key]

                            if isinstance(vid2['os'][key], list):
                                new_dis_video['os'][key] = new_os + vid2['os'][key]
                            else:
                                new_os.append(vid2['os'][key])
                                new_dis_video['os'][key] = new_os

                    newone.dis_videos.append(new_dis_video)
        return newone

    def to_combined_overlap_dataset_file(self, output_dataset_filepath, second_dataset_reader, **kwargs):
        combined_overlap_dataset = self.to_combined_overlap_dataset(second_dataset_reader, **kwargs)
        self.write_out_dataset(combined_overlap_dataset, output_dataset_filepath)

    def to_dictionary_style_dataset(self, **kwargs):
        if isinstance(self.dis_videos[0]['os'], dict):
            for dis_video in self.dis_videos:
                assert isinstance(dis_video['os'], dict), f"expect dis_video['os'] to be dict, but is: {dis_video['os']}"
            newone = self.dataset
        else:

            newone = self._prepare_new_dataset(kwargs)

            # ref_videos: deepcopy
            newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)

            # dis_videos: create a generic subject name for each entry in the list
            newone.dis_videos = []
            for dis_video in self.dis_videos:
                new_dis_video = copy.deepcopy(dis_video)
                new_dis_video['os'] = {}

                for idx, score in enumerate(dis_video['os']):
                    subj_name = self.dataset.dataset_name + '_subject' + str(idx)
                    new_dis_video['os'][subj_name] = score

                newone.dis_videos.append(new_dis_video)

        return newone

    def to_dictionary_style_dataset_file(self, dataset_filepath, **kwargs):
        dict_style_dataset = self.to_dictionary_style_dataset()
        self.write_out_dataset(dict_style_dataset, dataset_filepath)


class MockedRawDatasetReader(RawDatasetReader):

    def __init__(self, dataset, **kwargs):
        if 'input_dict' in kwargs:
            self.input_dict = kwargs['input_dict']
        else:
            self.input_dict = {}
        super().__init__(dataset)
        self._assert_input_dict()

    def to_dataset(self):
        """
        Override DatasetReader.to_dataset(). Need to overwrite dis_video['os']
        """

        newone = empty_object()
        newone.__dict__.update(self.dataset.__dict__)

        # deep copy ref_videos and dis_videos
        newone.ref_videos = copy.deepcopy(self.dataset.ref_videos)
        newone.dis_videos = copy.deepcopy(self.dis_videos)

        # overwrite dis_video['os']
        score_mtx = self.opinion_score_3darray
        num_videos, num_subjects, max_repetitions = score_mtx.shape
        assert num_videos == len(newone.dis_videos)
        for scores, dis_video in zip(score_mtx, newone.dis_videos):
            dis_video['os'] = list(scores)

        return newone


class SyntheticRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that generates synthetic data. It reads a dataset as baseline,
    and override the opinion_score_3darray based on input_dict.
    """

    @property
    def num_observers(self):
        assert 'observer_bias' in self.input_dict
        assert 'observer_inconsistency' in self.input_dict
        assert self.input_dict['observer_bias'].shape[0] == self.input_dict['observer_inconsistency'].shape[0]
        return self.input_dict['observer_bias'].shape[0]

    def _assert_input_dict(self):
        assert 'quality_scores' in self.input_dict
        assert 'observer_bias' in self.input_dict
        assert 'observer_inconsistency' in self.input_dict
        assert 'content_bias' in self.input_dict
        assert 'content_ambiguity' in self.input_dict

        E = len(self.input_dict['quality_scores'])
        S = len(self.input_dict['observer_bias'])
        C = len(self.input_dict['content_bias'])
        assert len(self.input_dict['observer_inconsistency']) == S
        assert len(self.input_dict['content_ambiguity']) == C

        assert E == self.num_dis_videos
        assert S == self.num_observers
        assert C == self.num_ref_videos

        if 'seed' in self.input_dict:
            assert self.input_dict['seed'] is None or isinstance(self.input_dict['seed'], int)

    @property
    def opinion_score_3darray(self):
        """
        Override DatasetReader.opinion_score_3darray(self), based on input
        synthetic_result.
        It follows the generative model:
        Z_e,s = Q_e + X_s + Y_[c(e)]
        where Q_e is the quality score of distorted video e, and X_s ~ N(b_s, sigma_s)
        is the term representing observer s's bias (b_s) and inconsistency (sigma_s).
        Y_c ~ N(mu_c, delta_c), where c is a function of e, or c = c(e), represents
        content c's bias (mu_c) and ambiguity (delta_c).
        """

        if 'seed' in self.input_dict:
            np.random.seed(self.input_dict['seed'])

        S = self.num_observers
        E = self.num_dis_videos

        q_e = np.array(self.input_dict['quality_scores'])
        q_es = np.tile(q_e, (S, 1)).T

        b_s = np.array(self.input_dict['observer_bias'])
        sigma_s = np.array(self.input_dict['observer_inconsistency'])
        x_es = np.tile(b_s, (E, 1)) + np.random.normal(0, 1, [E, S]) * np.tile(sigma_s, (E, 1))

        mu_c = np.array(self.input_dict['content_bias'])
        delta_c = np.array(self.input_dict['content_ambiguity'])
        mu_c_e = np.array(list(map(lambda i: mu_c[i], self.content_id_of_dis_videos)))
        delta_c_e = np.array(list(map(lambda i: delta_c[i], self.content_id_of_dis_videos)))
        y_es = np.tile(mu_c_e, (S, 1)).T + np.random.normal(0, 1, [E, S]) * np.tile(delta_c_e, (S, 1)).T

        z_es = q_es + x_es + y_es

        if 'quality_ambiguity' in self.input_dict:
            phi_e = np.array(self.input_dict['quality_ambiguity'])
            assert len(phi_e) == E
            z_es += np.random.normal(0, 1, [E, S]) * np.tile(phi_e, (S, 1)).T

        z_es_3d = np.zeros([E, S, 1])
        z_es_3d[:, :, 0] = z_es

        return z_es_3d


class SyntheticLogisticRawDatasetReader(SyntheticRawDatasetReader):

    @property
    def opinion_score_3darray(self):
        """
        Override DatasetReader.opinion_score_3darray(self), based on input
        synthetic_result.
        use logistic instead of
        """

        S = self.num_observers
        E = self.num_dis_videos

        q_e = np.array(self.input_dict['quality_scores'])
        q_es = np.tile(q_e, (S, 1)).T

        b_s = np.array(self.input_dict['observer_bias'])
        sigma_s = np.array(self.input_dict['observer_inconsistency'])
        x_es = np.tile(b_s, (E, 1)) + np.random.logistic(0, 1, [E, S]) * np.tile(sigma_s / (np.pi / np.sqrt(3)), (E, 1))

        mu_c = np.array(self.input_dict['content_bias'])
        delta_c = np.array(self.input_dict['content_ambiguity'])
        mu_c_e = np.array(list(map(lambda i: mu_c[i], self.content_id_of_dis_videos)))
        delta_c_e = np.array(list(map(lambda i: delta_c[i], self.content_id_of_dis_videos)))
        y_es = np.tile(mu_c_e, (S, 1)).T + np.random.logistic(0, 1, [E, S]) * np.tile(delta_c_e / (np.pi / np.sqrt(3)), (S, 1)).T

        z_es = q_es + x_es + y_es

        assert 'quality_ambiguity' not in self.input_dict

        z_es_3d = np.zeros([E, S, 1])
        z_es_3d[:, :, 0] = z_es

        return z_es_3d


class MissingDataRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that simulates random missing data. It reads a dataset as
    baseline, and override the opinion_score_3darray based on input_dict.
    """
    def _assert_input_dict(self):
        assert 'missing_probability' in self.input_dict

    @property
    def opinion_score_3darray(self):
        score_mtx = super(MissingDataRawDatasetReader, self).opinion_score_3darray

        if 'seed' in self.input_dict:
            np.random.seed(self.input_dict['seed'])
        mask = np.random.uniform(size=score_mtx.shape)
        mask[mask > self.input_dict['missing_probability']] = 1.0
        mask[mask <= self.input_dict['missing_probability']] = float('NaN')

        return score_mtx * mask


class SelectSubjectRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that only output selected subjects. It reads a dataset as a
    baseline, and override the opinion_score_3darray and other fields based on
    input_dict.
    """
    def _assert_input_dict(self):
        assert 'selected_subjects' in self.input_dict

        selected_subjects = self.input_dict['selected_subjects']

        # assert in 0, 1, 2...., num_observer -1
        observer_idxs = range(super(SelectSubjectRawDatasetReader, self).num_observers)
        for subject in selected_subjects:
            assert subject in observer_idxs

    @property
    def num_observers(self):
        return len(self.input_dict['selected_subjects'])

    @property
    def opinion_score_3darray(self):
        """
        3darray storing raw opinion scores, with first dimension the number of
        distorted videos, second dimension the number of observers, and third the maximum number of repetitions
        """
        selected_subjects = self.input_dict['selected_subjects']
        score_mtx = super().opinion_score_3darray
        score_mtx = score_mtx[:, selected_subjects, :]
        return score_mtx


class SelectDisVideoRawDatasetReader(MockedRawDatasetReader):

    def _assert_input_dict(self):
        assert 'selected_dis_videos' in self.input_dict

        selected_dis_videos = self.input_dict['selected_dis_videos']

        dis_video_idxs = range(len(super().dis_videos))
        for dis_video in selected_dis_videos:
            assert dis_video in dis_video_idxs

    @property
    @override(DatasetReader)
    def dis_videos(self):
        d_assetid_disvideo = dict()  # build dict: asset_id -> dis_video
        for dv in self.dataset.dis_videos:
            d_assetid_disvideo[dv['asset_id']] = dv
        return [d_assetid_disvideo[asset_id] for asset_id in self.input_dict['selected_dis_videos']]

    def to_dataset(self):
        raise NotImplementedError


class CorruptSubjectRawDatasetReader(MockedRawDatasetReader):
    """
    Dataset reader that have scores of selected subjects shuffled. It reads a
    dataset as a baseline, and override the opinion_score_3darray and other
    fields based on input_dict.
    """

    def _assert_input_dict(self):

        self._assert_selected_subjects()

        corrupt_behavior = self._get_corrupt_behavior()
        assert corrupt_behavior in ['shuffle', 'flip', 'min', 'mid', 'max', 'constant']

    def _assert_selected_subjects(self):
        assert 'selected_subjects' in self.input_dict
        selected_subjects = self.input_dict['selected_subjects']
        # assert no repeated numbers
        assert len(list(set(selected_subjects))) == len(selected_subjects)
        # assert in 0, 1, 2...., num_observer -1
        observer_idxs = range(super(CorruptSubjectRawDatasetReader, self).num_observers)
        for subject in selected_subjects:
            assert subject in observer_idxs

    @property
    def opinion_score_3darray(self):
        """
        3darray storing raw opinion scores, with first dimension the number of
        distorted videos, second dimension the number of observers, and third the maximum of repetitions
        """
        score_mtx = super(CorruptSubjectRawDatasetReader, self).opinion_score_3darray
        num_video, num_subject, max_repetitions = score_mtx.shape

        # for selected subjects, shuffle its score
        selected_subjects = self._get_selected_subjects()
        corrupt_probability = self._get_corrupt_probability()
        corrupt_behavior = self._get_corrupt_behavior()

        if corrupt_behavior == 'shuffle':

            for subject in selected_subjects:

                if corrupt_probability is not None:
                    videos = list(np.where(np.random.uniform(size=num_video) < corrupt_probability)[0])
                    score_mtx[videos, subject, :] = np.random.permutation(score_mtx[videos, subject, :])
                else:
                    np.random.shuffle(score_mtx[:, subject, :])

        elif corrupt_behavior == 'flip':

            min_score = np.nanmin(score_mtx)
            max_score = np.nanmax(score_mtx)

            for subject in selected_subjects:

                if corrupt_probability is not None:
                    videos = list(np.where(np.random.uniform(size=num_video) < corrupt_probability)[0])
                    score_mtx[videos, subject, :] = max_score + min_score - score_mtx[videos, subject, :]
                else:
                    score_mtx[:, subject, :] = max_score + min_score - score_mtx[:, subject, :]

        elif corrupt_behavior == 'min':

            min_score = np.nanmin(score_mtx)
            for subject in selected_subjects:

                if corrupt_probability is not None:
                    videos = list(np.where(np.random.uniform(size=num_video) < corrupt_probability)[0])
                    score_mtx[videos, subject, :] = min_score
                else:
                    score_mtx[:, subject, :] = min_score

        elif corrupt_behavior == 'mid':

            min_score = np.nanmin(score_mtx)
            max_score = np.nanmax(score_mtx)
            mid_score = (min_score + max_score) / 2.0

            for subject in selected_subjects:

                if corrupt_probability is not None:
                    videos = list(np.where(np.random.uniform(size=num_video) < corrupt_probability)[0])
                    score_mtx[videos, subject, :] = mid_score
                else:
                    score_mtx[:, subject, :] = mid_score

        elif corrupt_behavior == 'max':

            max_score = np.nanmax(score_mtx)
            for subject in selected_subjects:

                if corrupt_probability is not None:
                    videos = list(np.where(np.random.uniform(size=num_video) < corrupt_probability)[0])
                    score_mtx[videos, subject, :] = max_score
                else:
                    score_mtx[:, subject, :] = max_score

        elif corrupt_behavior == 'constant':

            min_score = np.nanmin(score_mtx)
            max_score = np.nanmax(score_mtx)

            for subject in selected_subjects:

                const_score = np.random.uniform(min_score, max_score)

                if corrupt_probability is not None:
                    videos = list(np.where(np.random.uniform(size=num_video) < corrupt_probability)[0])
                    score_mtx[videos, subject, :] = const_score
                else:
                    score_mtx[:, subject, :] = const_score

        else:
            assert False

        return score_mtx

    def _get_corrupt_probability(self):
        corrupt_probability = self.input_dict[
            'corrupt_probability'] if 'corrupt_probability' in self.input_dict else None
        return corrupt_probability

    def _get_selected_subjects(self):
        selected_subjects = self.input_dict['selected_subjects']
        return selected_subjects

    def _get_corrupt_behavior(self):
        corrupt_behavior = self.input_dict['corrupt_behavior'] \
            if 'corrupt_behavior' in self.input_dict and self.input_dict['corrupt_behavior'] is not None else 'shuffle'
        return corrupt_behavior


class CorruptDataRawDatasetReader(MockedRawDatasetReader):

    """
    Dataset reader that simulates random corrupted data. It reads a dataset as
    baseline, and override the opinion_score_3darray based on input_dict.
    """
    def _assert_input_dict(self):
        assert 'corrupt_probability' in self.input_dict

    @property
    def opinion_score_3darray(self):
        score_mtx = super(CorruptDataRawDatasetReader, self).opinion_score_3darray

        mask = np.random.uniform(size=score_mtx.shape)
        mask[mask > self.input_dict['corrupt_probability']] = 1.0
        mask[mask <= self.input_dict['corrupt_probability']] = float('NaN')

        score_mtx[np.isnan(mask)] = np.random.uniform(1, self.dataset.ref_score,
                                                      np.isnan(mask).sum())

        return score_mtx


class PairedCompDatasetReader(RawDatasetReader):
    """ Reader for a subjective quality test dataset with paired comparison scores. """

    def _assert_dataset(self):
        """
        Override RawDatasetReader._assert_dataset
        """
        super(PairedCompDatasetReader, self)._assert_dataset()

        num_dis_videos = self.num_dis_videos
        for dis_video in self.dis_videos:
            # e.g. 'os': {(' Diana Pena Alas', 120): 1, ...
            assert 'os' in dis_video
            assert isinstance(dis_video['os'], dict)
            for key in dis_video['os'].keys():
                assert isinstance(key[0], str)
                assert isinstance(key[1], int)
            # for now, asset_id must be continuous
            assert dis_video['asset_id'] >= 0 and dis_video['asset_id'] < num_dis_videos, \
                'asset_id must be in [0, {}) but is {}'.format(num_dis_videos, dis_video['asset_id'])

    @property
    @persist
    def opinion_score_3darray(self):
        """ 3darray storing raw opinion scores, with first dimension the distorted videos (PVS),
        second dimension the distorted videos (PVS) compared against, and third dimension the
        observers. """

        list_observers = self._get_list_observers()

        # build dict: observer -> i_observer
        dict_observer_to_iobserver = dict()
        for i_observer, observer in enumerate(list_observers):
            dict_observer_to_iobserver[observer] = i_observer

        score_3darray = float("NaN") * np.ones([self.num_dis_videos, self.num_dis_videos, self.num_observers])

        for i_dis_video, dis_video in enumerate(self.dis_videos):
            for key, value in dis_video['os'].items():
                assert value in [0, 0.5, 1], \
                    f"expect value in [0, 0.5, 1], but got: {value}"
                subject, pvs_j = key
                pvs_i = i_dis_video
                score_3darray[pvs_i][pvs_j][dict_observer_to_iobserver[subject]] = value  # CAUTION: note the dimension change!
                score_3darray[pvs_j][pvs_i][dict_observer_to_iobserver[subject]] = 1.0 - value

        return score_3darray

    def _get_list_observers(self):
        for dis_video in self.dis_videos:
            assert isinstance(dis_video['os'], dict)

        list_observers = []
        for dis_video in self.dis_videos:
            observers = map(lambda x: x[0], dis_video['os'].keys())
            observers = list(set(observers))
            list_observers += observers

        return get_unique_sorted_list(list_observers)

    @property
    def ref_score(self):
        raise NotImplementedError

    def to_persubject_dataset(self, quality_scores, **kwargs):
        raise NotImplementedError

    def to_combined_overlap_dataset(self, second_dataset_filepath, **kwargs):
        raise NotImplementedError
