# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for datasets."""

import functools
import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from coltran.utils import datasets_utils
import glob
import cv2
import random
from absl import logging

def resize_to_square(image, resolution=32, train=True):
  """Preprocess the image in a way that is OK for generative modeling."""

  # Crop a square-shaped image by shortening the longer side.
  image_shape = tf.shape(image)
  height, width, channels = image_shape[0], image_shape[1], image_shape[2]
  if height == width:
    side_size = resolution
  else:
    side_size = tf.minimum(height, width) 
  cropped_shape = tf.stack([side_size, side_size, channels])
  if train:
    image = tf.image.random_crop(image, cropped_shape)
  else:
    image = tf.image.resize_with_crop_or_pad(
        image, target_height=side_size, target_width=side_size)

  image = datasets_utils.change_resolution(image, res=resolution, method='area')
  return image


def preprocess(example, train=True, resolution=256):
  """Apply random crop (or) central crop to the image."""
  image = example

  is_label = False
  mask = None
  if isinstance(example, dict):
    image = example['image']
    is_label = 'label' in example.keys()
    mask = tf.cast(example['mask'], tf.uint8)

  if mask is not None:
    ind_mask = image.shape[-1]
    image = tf.concat((image, mask), axis=2)

  image = resize_to_square(image, train=train, resolution=resolution)

  # keepng 'file_name' key creates some undebuggable TPU Error.
  example_copy = dict()
  if mask is not None:
    image, mask = tf.split(image, [ind_mask, -1], axis=2)
    targets = image * tf.keras.backend.repeat_elements(mask, 3, axis=2)
    # mask = tf.cast(mask, tf.bool)
    example_copy['mask'] = mask
  else:
    targets = image

  example_copy['image'] = image
  example_copy['targets'] = targets
  if is_label:
    example_copy['label'] = example['label']

  return example_copy


def get_gen_dataset(data_dir, batch_size):
  """Converts a list of generated TFRecords into a TF Dataset."""

  def parse_example(example_proto, res=64):
    features = {'image': tf.io.FixedLenFeature([res*res*3], tf.int64)}
    example = tf.io.parse_example(example_proto, features=features)
    image = tf.reshape(example['image'], (res, res, 3))
    return {'targets': image}

  # Provided generated dataset.
  def tf_record_name_to_num(x):
    x = x.split('.')[0]
    x = re.split(r'(\d+)', x)
    return int(x[1])

  assert data_dir is not None
  records = tf.io.gfile.listdir(data_dir)
  max_num = max(records, key=tf_record_name_to_num)
  max_num = tf_record_name_to_num(max_num)

  records = []
  for record in range(max_num + 1):
    path = os.path.join(data_dir, f'gen{record}.tfrecords')
    records.append(path)

  tf_dataset = tf.data.TFRecordDataset(records)
  tf_dataset = tf_dataset.map(parse_example, num_parallel_calls=100)
  tf_dataset = tf_dataset.batch(batch_size=batch_size)
  tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
  return tf_dataset


def create_gen_dataset_from_images(image_dir, mask_dir, train):
  """Creates a dataset from the provided directory."""
  def load_image(path):
    image_str = tf.io.read_file(path)
    return tf.image.decode_image(image_str, channels=3)

  def categorize_mask(mask):
    mask = cv2.imread(mask, 0)
    coverage = (np.count_nonzero(mask) / mask.size) * 100.0
    if coverage <= 5.0:
      category = 'clear'
    elif 5.0 < coverage <= 30.0:
      category = 'moderate'
    elif coverage > 30.0:
      category = 'severe'
    return category

  mod_masks = []
  files = []
  cube = []
  cloud_masks = []
  hypercube = []
  masks_hcube = []

  for r, m in zip(sorted(glob.glob(image_dir + "/**")), sorted(glob.glob(mask_dir + "/**"))):
    for im, mask, ind in zip(sorted(glob.glob(r + "/**")), sorted(glob.glob(m + "/**")), enumerate(sorted(glob.glob(r + "/**")))):

      # create a list with moderately cloudy masks to use later
      category = categorize_mask(mask)
      if category == 'moderate':
        mod_masks.append(mask)

      # create the cube with the (T-4, ..., T-1) images masked with their own masks
      if 51 <= ind[0] < 55:  # FIXME: Hard-coded indexes
        im_mask = cv2.imread(im)
        curr_mask = cv2.imread(mask, 0)
        cloud_masks.append(curr_mask == 0)
        # im_mask[curr_mask > 0] = 0
        # im_mask[curr_mask == 0] = im_mask[curr_mask == 0]
        im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2RGB)
        cube.append(im_mask)  # encoder's input

      # add the last image to the cube list 2 times, with a random moderate mask + as is
      elif ind[0] == 55:  # FIXME: Hard-coded indexes
        im_mask = cv2.imread(im)
        curr_mask = cv2.imread(mask, 0)
        cloud_masks.append(curr_mask == 0)
        # mask the last image with its own mask for further masking later
        # im_mask[curr_mask > 0] = 0
        # im_mask[curr_mask == 0] = im_mask[curr_mask == 0]

        # decoder's input
        last_clear = load_image(im)
        # cube.append(last_clear)
        cube.insert(0, last_clear)
        cloud_masks.insert(0, curr_mask == 0)

    # if there is no moderate cloudy mask skip the area
    if len(mod_masks) == 0:
      cube.clear()
      cloud_masks.clear()
      logging.info("Not enough moderate cloudy masks found, skipping")
      continue

    # random masking on the last image
    if train:
      curr_mask = cv2.imread(random.choice(mod_masks), 0)
    else:
      curr_mask = cv2.imread(mod_masks[0], 0)
    cloud_masks[-1] *= curr_mask == 0
    # im_mask[curr_mask > 0] = 0
    # im_mask[curr_mask == 0] = im_mask[curr_mask == 0]
    im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2RGB)
    cube.append(im_mask)  # encoder's input

    # apply random masking on every image during training
    if train:
      for i in range(len(cube) - 1):
        curr_mask = cv2.imread(random.choice(mod_masks), 0)
        cloud_masks[i+1] *= curr_mask == 0
        img = cube[i+1]
        # img[curr_mask > 0] = 0
        # img[curr_mask == 0] = img[curr_mask == 0]
        cube[i + 1] = img
    
    # if the last image is not clear skip the area
    last_category = categorize_mask(mask)
    if train and last_category != 'clear':
      cube.clear()
      cloud_masks.clear()
      continue

    # save the randomly cloudy image for evaluation
    # gen = Image.fromarray(cube[-1], mode='RGB')
    # if train:
    #   gen.save('D:\\Timeseries_cropped_512\\gen_for_eval\\train\\' + os.path.basename(r) + '.jpeg')
    # else:
    #   gen.save('D:\\Timeseries_cropped_512\\gen_for_eval\\test\\' + os.path.basename(r) + '.jpeg')

    files = tf.concat(cube, axis=2)  # creates tensors with size (256,256,T*3)
    cube_masks = tf.stack(cloud_masks, axis=2)

    yield {'image': files, 'mask': cube_masks}
    # # print(files)
    # hypercube.append(files)  # list of (N) selected tensors with size (256,256,T*3)
    # masks_hcube.append(cube_masks)
    # # print(hypercube)
    cube.clear()
    cloud_masks.clear()
    
    # # print(mod_masks)
    # mod_masks.clear()
  
  #print(len(hypercube))
  # dataset = tf.data.Dataset.from_tensor_slices({'image': (hypercube), 'mask': (masks_hcube)})
  # return dataset


def get_imagenet(subset, read_config):
  """Gets imagenet dataset."""
  train = subset == 'train'
  num_val_examples = 0 if subset == 'eval_train' else 20000
  if subset == 'test':
    ds = tfds.load('imagenet2012', split='validation', shuffle_files=False, data_dir='D:\\tensorflow_datasets')
  else:
    # split 10000 samples from the imagenet dataset for validation.
    ds, info = tfds.load('imagenet2012', split='train', with_info=True,
                         shuffle_files=train, read_config=read_config, data_dir='D:\\tensorflow_datasets')
    num_train = info.splits['train'].num_examples - num_val_examples
    if train:
      ds = ds.take(num_train)
    elif subset == 'valid':
      ds = ds.take(num_val_examples)
  return ds


def get_dataset(name,
                config,
                batch_size,
                subset,
                read_config=None,
                data_dir=None):
  """Wrapper around TF-Datasets.

  * Setting `config.random_channel to be True` adds
    ds['targets_slice'] - Channel picked at random. (of 3).
    ds['channel_index'] - Index of the randomly picked channel
  * Setting `config.downsample` to be True, adds:.
    ds['targets_64'] - Downsampled 64x64 input using tf.resize.
    ds['targets_64_up_back] - 'targets_64' upsampled using tf.resize

  Args:
    name: imagenet
    config: dict
    batch_size: batch size.
    subset: 'train', 'eval_train', 'valid' or 'test'.
    read_config: optional, tfds.ReadConfg instance. This is used for sharding
                 across multiple workers.
    data_dir: Data Directory, Used for Custom dataset.
  Returns:
   dataset: TF Dataset.
  """
  downsample = config.get('downsample', False)
  random_channel = config.get('random_channel', False)
  downsample_res = config.get('downsample_res', 64)
  downsample_method = config.get('downsample_method', 'area')
  num_epochs = config.get('num_epochs', -1)
  data_dir = config.get('data_dir') or data_dir
  mask_dir = config.get('mask_dir')
  auto = tf.data.AUTOTUNE
  train = subset == 'train'

  if name == 'imagenet':
    ds = get_imagenet(subset, read_config)
  elif name == 'custom':
    assert data_dir is not None
    ds = tf.data.Dataset.from_generator(
              lambda: create_gen_dataset_from_images(data_dir, mask_dir, train=train),
              output_signature={'image': tf.TensorSpec(shape=(None,None,config.timeline*3), dtype=tf.uint8),
                                'mask': tf.TensorSpec(shape=(None,None,config.timeline), dtype=tf.bool)}
          )
  else:
    raise ValueError(f'Expected dataset in [imagenet, custom]. Got {name}')

  ds = ds.map(
      lambda x: preprocess(x, train=train), num_parallel_calls=100)
  if train and random_channel:
    ds = ds.map(datasets_utils.random_channel_slice)
  if downsample:
    downsample_part = functools.partial(
        datasets_utils.downsample_and_upsample,
        train=train,
        downsample_res=downsample_res,
        upsample_res=256,
        method=downsample_method)
    ds = ds.map(downsample_part, num_parallel_calls=100)

  # datasets_utils.save_dataset(ds, config.get('targets_dir'), train)
  
  if train:
    ds = ds.repeat(num_epochs)
    ds = ds.shuffle(buffer_size=128)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(auto)
  print(ds) #compare with imagenet
  return ds
