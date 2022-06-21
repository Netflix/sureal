# test dataset with opinion scores (os) in dictionary style

dataset_name = 'test_dataset_os_as_dict2'
yuv_fmt = 'yuv420p'
width = 1920
height = 1080
ref_score = 5.0

ref_videos = [
       {'content_id': 0, 'content_name': 'foo', 'path': 'foo.png'},
       {'content_id': 1, 'content_name': 'bar_2', 'path': 'bar_2.png'}
]

dis_videos = [
              {'asset_id': 0,
               'content_id': 0,
               'os': {'Tom': 3, 'Merry': 4, 'Pipin': 1},
               'path': 'baz1.png'},
              {'asset_id': 1,
               'content_id': 1,
               'os': {'Tom': 2, 'Merry': 1, 'Pipin': 3},
               'path': 'baz2_2.png'},
              {'asset_id': 2,
               'content_id': 0,
               'os': {'Tom': 4, 'Merry': 1, 'Pipin': 3},
               'path': 'baz3.png'}
]
