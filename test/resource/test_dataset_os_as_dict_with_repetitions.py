# test dataset with opinion scores (os) in dictionary style with repetitions

dataset_name = 'test_dataset_os_as_dict_with_repetitions'
yuv_fmt = 'yuv420p'
width = 1920
height = 1080
ref_score = 5.0

ref_videos = [
       {'content_id': 0, 'content_name': 'foo', 'path': 'foo.png'},
       {'content_id': 1, 'content_name': 'bar', 'path': 'bar.png'}
]

dis_videos = [
              {'asset_id': 0,
               'content_id': 0,
               'os': {'Tom': [3, 2], 'Jerry': 4, 'Pinokio': 1},
               'path': 'baz1.png'},
              {'asset_id': 1,
               'content_id': 1,
               'os': {'Tom': [2, 2], 'Jerry': 1, 'Pinokio': [3, 3, 1]},
               'path': 'baz2.png'},
              {'asset_id': 2,
               'content_id': 0,
               'os': {'Tom': 4, 'Jerry': 1, 'Pinokio': 3},
               'path': 'baz3.png'}
]
