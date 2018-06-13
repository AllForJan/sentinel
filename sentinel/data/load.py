# coding: utf-8
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import tifffile

from .write import get_file

## LOAD TF DATASET ##
def parse_sizes(parsed_features):
  h = tf.cast(parsed_features['image_height'], tf.int32)
  w = tf.cast(parsed_features['image_width'], tf.int32)
  return h, w


def decode_image(parsed, shape):
  image = tf.decode_raw(parsed, tf.int32)
  return tf.reshape(image, tf.stack(shape))


def parse_example(proto, sparse):
  """The image is already offsetted.
  
  [description]
  
  Arguments:
    example_proto {[type]} -- [description]
  
  Returns:
    [type] -- [description]
  """

  if sparse:
    features = {
        'image_height': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image_width': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image': tf.FixedLenFeature([], tf.string, default_value=""),
        'mask_indices': tf.FixedLenFeature([], tf.string, default_value=""),
        'mask_values': tf.FixedLenFeature([], tf.string, default_value=""),
    }
  else:
    features = {
        'image_height': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image_width': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image': tf.FixedLenFeature([], tf.string, default_value=""),
        'mask': tf.FixedLenFeature([], tf.string, default_value=""),
    }

  parsed_features = tf.parse_single_example(proto, features)

  h, w = parse_sizes(parsed_features)
  image = decode_image(parsed_features['image'], (h, w, 3))

  if sparse:
    indices = decode_image(parsed_features['mask_indices'], (-1, 4))
    indices = tf.to_int64(indices)
    values = decode_image(parsed_features['mask_values'], (-1,))
    mask = tf.sparse_to_dense(
        sparse_indices=indices,
        output_shape=tf.to_int64([1, h, w]),  # bug
        sparse_values=values,
        default_value=0,
        validate_indices=True,
        name=None)
  else:
    mask = decode_image(parsed_features['mask'], (h, w))

  return tf.to_float(image), tf.to_float(mask)


def parse_tfrecord(example_proto, sparse, image_height=None, image_width=None):
  image, mask = parse_example(example_proto, sparse)

  # instead of using padded_batch use auto-cropping or - padding
  if image_height is not None and image_width is not None:
    resize = lambda x: tf.image.resize_image_with_crop_or_pad(
        image=x, target_height=image_height, target_width=image_width)
    image, mask = resize(image), resize(mask)

  return image, mask


def get_patches(image, mask, num_patches=100, patch_size=16):
  """Get `num_patches` random crops from the image"""
  tensor = tf.concat([image, mask[..., None]], -1)  # (h, w, 4)
  patches = []
  for i in range(num_patches):
    patch = tf.random_crop(tensor, [patch_size, patch_size, 4])
    patches.append(patch)
  patches = tf.stack(patches)
  assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 4]
  return patches[..., :3], patches[..., 3:]


def TrainSampleParseFn(num_patches, patch_size):
  buffer_size = 50 * num_patches
  num_parallel_calls = 4 * num_patches
  sampler_parse_fn = lambda y: get_patches(
    *parse_tfrecord(y, sparse=False, image_height=None, image_width=None),
    num_patches=num_patches, patch_size=patch_size)
  return lambda x: (tf.data.TFRecordDataset(x)
                    .shuffle(buffer_size=buffer_size)
                    .map(sampler_parse_fn, num_parallel_calls=num_parallel_calls)
                    .apply(tf.contrib.data.unbatch()))


def TestSampleParseFn(num_parallel_calls):
  parse_fn = lambda y: parse_tfrecord(
    y, sparse=False, image_height=None, image_width=None)
  return lambda x: (tf.data.TFRecordDataset(x)
                    .map(parse_fn, num_parallel_calls=num_parallel_calls))


def input_fn(files, num_patches, patch_size, batch_size, train):
  if train:
    parse_fn = TrainSampleParseFn(num_patches, patch_size)
  else:
    parse_fn = TestSampleParseFn(batch_size)
  dataset = create_dataset(files, parse_fn)
  if train:
    dataset = dataset.shuffle(buffer_size=50 * num_patches)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
  else:
    padded_shapes = ((None, None, 3), (None, None))
    dataset = dataset.padded_batch(batch_size=1, padded_shapes=padded_shapes)
  dataset = dataset.prefetch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def create_dataset(filenames, parse_fn):
  apply_func = tf.contrib.data.parallel_interleave(
      parse_fn, cycle_length=len(filenames), block_length=len(filenames))
  return tf.data.Dataset.from_tensor_slices(filenames).apply(apply_func)


def get_input_fn(filenames, num_patches, patch_size, batch_size, test_size=.33):
  train_files, test_files = train_test_split(
      filenames, shuffle=True, test_size=test_size)
  eval_files, test_files = train_test_split(
      test_files, shuffle=True, test_size=test_size)

  def train_input_fn():
    return input_fn(
        train_files, num_patches, patch_size, batch_size, train=True)

  def eval_input_fn():
    return input_fn(
        eval_files, num_patches, patch_size, batch_size, train=False)

  return train_input_fn, eval_input_fn


## IMAGES ##
def load_band_images(file_pattern='three_band/*'):
  rgb_bands = sorted(glob(get_file(file_pattern)))
  return [tifffile.imread(band_file) for band_file in rgb_bands]


def read_image_files(file_pattern='three_band/*'):
  rgb_bands = sorted(glob(get_file(file_pattern)))
  images_df = pd.DataFrame(rgb_bands, columns=['image_path'])
  split = lambda x: x.split('/')[-1]
  images_df['image_id'] = images_df.image_path.str.strip('.tif').apply(split)
  return images_df


def stretch_8bit(bands, lower_percent=2, higher_percent=98):
  # https://www.kaggle.com/aamaia/rgb-using-m-bands-example/code
  out = np.zeros_like(bands)
  for i in range(3):
    a = 0
    b = 255
    c = np.percentile(bands[:, :, i], lower_percent)
    d = np.percentile(bands[:, :, i], higher_percent)
    t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
    t[t < a] = a
    t[t > b] = b
    out[:, :, i] = t
  return out.astype(np.uint8)


def read_image(image_id, file_pattern='sixteen_band/{}_M.tif'):
  filename = get_file(file_pattern.format(image_id))
  img = tifffile.imread(filename)
  img = np.rollaxis(img, 0, 3)
  return img
