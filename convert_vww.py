import tensorflow as tf

import tensorflow_datasets as tfds

# converting to tfds by generating the metadata

features = tfds.features.FeaturesDict({
    'image/encoded': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
    'image/class/label': tfds.features.ClassLabel(names=['background', 'person']),
    # 'image/object/bbox': tfds.features.BBox(),
})

split_info = tfds.folder_dataset.compute_split_info(
    out_dir = 'gs://wake-vision/vww/',
    filename_template=tfds.core.ShardedFileTemplate('gs://wake-vision/vww/',
                        '{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}', 'vww'),
)

tfds.folder_dataset.write_metadata(
    data_dir='gs://wake-vision/vww/',
    features=features,
    # Pass the `out_dir` argument of compute_split_info (see section above)
    # You can also explicitly pass a list of `tfds.core.SplitInfo`.
    split_infos=split_info,
    # Pass a custom file name template or use None for the default TFDS
    # file name template.
    filename_template=tfds.core.ShardedFileTemplate('gs://wake-vision/vww/',
                        '{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}', 'vww'),

    # Optionally, additional DatasetInfo metadata can be provided
    # See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    # description="""Multi-line description."""
    # homepage='http://my-project.org',
    supervised_keys=('image', 'label'),
)