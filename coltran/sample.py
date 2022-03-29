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

"""ColTran: Sampling scripts."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import numpy as np
from PIL import Image, ImageEnhance

import tensorflow.compat.v2 as tf
from werkzeug.debug import console

from coltran import datasets
from coltran.models import colorizer
from coltran.models import upsampler
from coltran.utils import base_utils
from coltran.utils import datasets_utils
from coltran.utils import train_utils

from matplotlib import cm

# pylint: disable=g-direct-tensorflow-import

# pylint: disable=missing-docstring
# pylint: disable=not-callable
# pylint: disable=g-long-lambda

flags.DEFINE_enum('mode', 'sample_test', [
    'sample_valid', 'sample_test', 'sample_train'], 'Operation mode.')

flags.DEFINE_string('logdir', '/tmp/svt', 'Main directory for logs.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU', 'TPU'],
                  'Hardware type.')
flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_string('summaries_log_dir', 'summaries', 'Summaries parent.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('devices_per_worker', 1, 'Number of devices per worker.')
flags.DEFINE_integer('num_workers', 1, 'Number workers.')
config_flags.DEFINE_config_file(
    'config',
    default='test_configs/colorizer.py',
    help_string='Training configuration file.')

FLAGS = flags.FLAGS


def array_to_tf_example(array, label):
  """Converts array to a serialized TFExample."""
  array = np.ravel(array)
  x_list = tf.train.Int64List(value=array)
  label_list = tf.train.Int64List(value=np.array([label]))
  feature_dict = {
      'image': tf.train.Feature(int64_list=x_list),
      'label': tf.train.Feature(int64_list=label_list),
  }
  x_feats = tf.train.Features(feature=feature_dict)
  example = tf.train.Example(features=x_feats)
  return example.SerializeToString()


def build(config, batch_size, is_train=False):
  optimizer = train_utils.build_optimizer(config)
  ema_vars = []

  downsample = config.get('downsample', False)
  downsample_res = config.get('downsample_res', 64)
  h, w = config.resolution

  if config.model.name == 'coltran_core':
    if downsample:
      h, w = downsample_res, downsample_res
    zero_slice = tf.zeros((batch_size, h, w, config.get('timeline', 6)), dtype=tf.int32)
    zero = tf.zeros((batch_size, h, w, 3*config.get('timeline', 6)), dtype=tf.int32)
    model = colorizer.ColTranCore(config.model)
    model(zero, inputs_slice=zero_slice, channel_index=0, training=is_train)

  c = 1 if is_train else 3
  if config.model.name == 'color_upsampler':
    if downsample:
      h, w = downsample_res, downsample_res
    zero_slice = tf.zeros((batch_size, h, w, c), dtype=tf.int32)
    zero = tf.zeros((batch_size, h, w, 3*config.get('timeline', 6)), dtype=tf.int32)
    model = upsampler.ColorUpsampler(config.model)
    model(zero, inputs_slice=zero_slice, training=is_train)
  elif config.model.name == 'spatial_upsampler':
    zero_slice = tf.zeros((batch_size, h, w, c), dtype=tf.int32)
    zero = tf.zeros((batch_size, h, w, 3*config.get('timeline', 6)), dtype=tf.int32)
    model = upsampler.SpatialUpsampler(config.model)
    model(zero, inputs_slice=zero_slice, training=is_train)

  ema_vars = model.trainable_variables
  ema = train_utils.build_ema(config, ema_vars)
  return model, optimizer, ema


def get_grayscale_at_sample_time(data, downsample_res, model_name):
  if model_name == 'spatial_upsampler':
    curr_rgb = data['targets']
  else:
    curr_rgb = data['targets_%d' % downsample_res]
  return curr_rgb


def create_sample_dir(logdir, config):
  """Creates child directory to write samples based on step name."""
  sample_dir = config.sample.get('log_dir')
  if config.sample.only_parallel:
    sample_dir += "_parallel"
  assert sample_dir is not None
  sample_dir = os.path.join(logdir, sample_dir)
  tf.io.gfile.makedirs(sample_dir)
  logging.info('writing samples at: %s', sample_dir)
  return sample_dir

def store_samples(data, config, logdir, subset, gen_dataset=None):
  def nearest_upsample(x, factor=1):
    if factor > 1:
      x = x.repeat(factor, axis=0).repeat(factor, axis=1)
    return x

  def im_after_proc(x, mode, upsample_factor=4, brightness_factor=1.5, contrast_factor=0.8): 
      im = Image.fromarray(nearest_upsample(x, upsample_factor), mode=mode)
      if mode == 'RGB':
        im = ImageEnhance.Brightness(im).enhance(brightness_factor)
        im = ImageEnhance.Contrast(im).enhance(contrast_factor)
      return im

  """Stores the generated samples."""
  downsample_res = config.get('downsample_res', 64)
  num_samples = config.sample.num_samples
  num_outputs = config.sample.num_outputs
  batch_size = config.sample.get('batch_size', 1)
  sample_mode = config.sample.get('mode', 'argmax')
  gen_file = config.sample.get('gen_file', 'gen')
  cmap_name = config.sample.get('cmap', 'inferno')
  up_factor = config.sample.get('upsample_factor', 4)

  colmap = cm.get_cmap(cmap_name)(np.linspace(0, 1, config.timeline))
  colmapint = np.round(colmap[...,:3]*255.).flatten().astype(np.uint8)

  model, optimizer, ema = build(config, 1, False)
  checkpoints = train_utils.create_checkpoint(model, optimizer, ema)
  sample_dir = create_sample_dir(logdir, config)
  record_path = os.path.join(sample_dir, '%s.tfrecords' % gen_file)
  writer = tf.io.TFRecordWriter(record_path)

  train_utils.restore(model, checkpoints, logdir, ema)
  num_steps_v = optimizer.iterations.numpy()
  logging.info('Producing sample after %d training steps.', num_steps_v)

  sample_summary_dir = os.path.join(
      logdir, 'sample_{}'.format(subset))
  writer_smr = tf.summary.create_file_writer(sample_summary_dir)

  psnr_vals = np.ones((num_outputs//batch_size*batch_size, num_samples))*np.nan
  ssim_vals = np.ones((num_outputs//batch_size*batch_size, num_samples))*np.nan
  mse_vals = np.ones((num_outputs//batch_size*batch_size, num_samples))*np.nan
  logging.info(gen_dataset)
  for batch_ind in range(num_outputs // batch_size):
    batch_ind = batch_ind + config.sample.skip_batches*batch_size
    next_data = data.next()
    labels = tf.zeros((batch_size,), dtype=tf.int32).numpy()
    #labels = next_data['label'].numpy()

    if gen_dataset is not None:
      next_gen_data = gen_dataset.next()

    # Gets grayscale image based on the model.
    curr_gray = get_grayscale_at_sample_time(next_data, downsample_res,
                                             config.model.name)

    curr_output = collections.defaultdict(list)
    for sample_ind in range(num_samples):
      logging.info('Batch no: %d, Sample no: %d', batch_ind, sample_ind)

      if config.model.name == 'color_upsampler':

        if gen_dataset is not None:
          # Provide generated coarse color inputs.
          scaled_rgb = next_gen_data['targets']
        else:
          # Provide coarse color ground truth inputs.
          scaled_rgb = next_data['targets_%d' % downsample_res]
        bit_rgb = base_utils.convert_bits(scaled_rgb, n_bits_in=8, n_bits_out=3)
        output = model.sample(gray_cond=curr_gray, bit_cond=bit_rgb,
                              mode=sample_mode)

      elif config.model.name == 'spatial_upsampler':
        if gen_dataset is not None:
          # Provide low resolution generated image.
          low_res = next_gen_data['targets']
          low_res = datasets_utils.change_resolution(low_res, 256)
        else:
          # Provide low resolution ground truth image.
          low_res = next_data['targets_%d_up_back' % downsample_res]
        output = model.sample(gray_cond=curr_gray, inputs=low_res,
                              mode=sample_mode)
      else:
        output = model.sample(gray_cond=curr_gray, mode=sample_mode, 
                                  only_parallel=config.sample.only_parallel)
      logging.info('Done sampling')

      #current = curr_gray[:, :, :, :3]
      # check differences in pixel values between the ground truth and the generated sample image
      #result = tf.unique_with_counts(tf.reshape(tf.abs(current[:] - tf.cast(output['bit_up_argmax'][:], tf.int32)), [-1]))
      # check differences in pixel values between the ground truth and the generated sample image in the area of masks
      #mask_result = tf.unique_with_counts(tf.reshape(tf.abs(current[curr_gray[:, :, :, -3:] == [0, 0, 0]] - tf.cast(output['bit_up_argmax'][curr_gray[:, :, :, -3:] == [0, 0, 0]], tf.int32)), [-1]))

      #print(result)
      #print(mask_result)

      for out_key, out_item in output.items():
        curr_output[out_key].append(out_item.numpy())


    input_cube = tf.cast(curr_gray, dtype=tf.uint8)
    current = input_cube[..., :3]
    current_comp = tf.cast(current, dtype=tf.uint8)
    cube = tf.reshape(input_cube[..., 3:], list(input_cube.shape[0:3]) + [-1, 3])
    cube = tf.transpose(cube, [3,1,2,4,0])
    sample_key = None
    for out_key, out_val in output.items():
      if ('sample' in out_key or 'argmax' in out_key) or \
              ('core' in config.model.name and config.sample.only_parallel and 'parallel' in out_key):
        sample_key = out_key
        output_samples = np.concatenate(curr_output[sample_key], axis=0).astype(np.uint8)
        break
    if sample_key:
      for local_ind in range(batch_size):
        output_ind = batch_ind*batch_size + local_ind
        gen_im = output_samples[local_ind::batch_size,...]
        ref_im = current_comp[local_ind,...]
        psnr_vals[output_ind, :] = tf.image.psnr(output_samples[local_ind::batch_size,...], 
                                                  ref_im, 255)
        ssim_vals[output_ind, :] = tf.image.ssim(output_samples[local_ind::batch_size,...], 
                                                  ref_im, 255)
        mse_vals[output_ind, :] = tf.reduce_mean(tf.metrics.mse(tf.cast(gen_im, tf.float32), tf.cast(ref_im, tf.float32)))
        logging.info("Output %d, PSNR: %.4f/%.4f, SSIM: %.4f/%.4f, MSE: %.4f/%.4f", output_ind,
              np.average(psnr_vals[output_ind, :]), np.std(psnr_vals[output_ind, :]),
              np.average(ssim_vals[output_ind, :]), np.std(ssim_vals[output_ind, :]),
              np.average(mse_vals[output_ind, :]), np.std(mse_vals[output_ind, :]))

        with writer_smr.as_default():          
          tf.summary.scalar('PSNR'+sample_key, np.average(psnr_vals[output_ind, :]), step=output_ind)
          tf.summary.scalar('SSIM'+sample_key, np.average(ssim_vals[output_ind, :]), step=output_ind)
          tf.summary.scalar('MSE'+sample_key, np.average(mse_vals[output_ind, :]), step=output_ind)
          tf.summary.image('GT'+sample_key, current[local_ind:local_ind+1,...], step=output_ind)
          tf.summary.image('Input'+sample_key, cube[::-1,:,:,:,local_ind], step=output_ind, 
                            max_outputs=1)
          tf.summary.image('Samples'+sample_key, output_samples[local_ind::batch_size,...], step=output_ind)

        if config.sample.im_outputs:          
          unmasked = next_data[f'image_{downsample_res}'][local_ind,...,:3].numpy()
          unmasked_im = im_after_proc(unmasked.astype(np.uint8), mode='RGB')
          gt_im = im_after_proc(nearest_upsample(current[local_ind,...].numpy(), up_factor), 
          mode='RGB')
          target_im = im_after_proc(cube[-1,:,:,:,local_ind].numpy(), mode='RGB')
          sample_im = im_after_proc(output_samples[local_ind,...], mode='RGB')
          cover_mask = np.sum(next_data[f'mask_{downsample_res}'][local_ind,...,1:], axis=2) #np.sum(np.any(cube[::-1,:,:,:,local_ind]!=0, axis=3),axis=0)
          cover_mask_im = im_after_proc(cover_mask.astype(np.uint8), mode='P')
          cover_mask_im.putpalette(colmapint)
          unmasked_im.save(os.path.join(sample_dir, f'{output_ind:05d}_unm.png'))
          gt_im.save(os.path.join(sample_dir, f'{output_ind:05d}_gt.png'))
          target_im.save(os.path.join(sample_dir, f'{output_ind:05d}_tar.png'))
          sample_im.save(os.path.join(sample_dir, f'{output_ind:05d}_sam.png'))
          cover_mask_im.save(os.path.join(sample_dir, f'{output_ind:05d}_cov.png'))  

    # concatenate samples across width.
    for out_key, out_val in curr_output.items():
      curr_out_val = np.concatenate(out_val, axis=2)
      curr_output[out_key] = curr_out_val

      if ('sample' in out_key or 'argmax' in out_key) or \
              ('core' in config.model.name and config.sample.only_parallel and 'parallel' in out_key):
        save_str = f'Saving {(batch_ind + 1) * batch_size} samples'
        logging.info(save_str)
        for single_ex, label in zip(curr_out_val, labels):
          serialized = array_to_tf_example(single_ex, label)
          writer.write(serialized)

  std_mean = lambda arr: np.sqrt(np.sum(np.var(arr, axis=1)))/arr.shape[0]
  logging.info("Average PSNR: %.4f/%.4f, Average SSIM: %.4f/%.4f, Average MSE: %.4f/%.4f", 
                np.average(psnr_vals), std_mean(psnr_vals),
                np.average(ssim_vals), std_mean(ssim_vals),
                np.average(mse_vals), std_mean(mse_vals))
  writer.close()


def sample(logdir, subset):
  """Executes the sampling loop."""
  logging.info('Beginning sampling loop...')
  config = FLAGS.config
  batch_size = config.sample.get('batch_size', 1)
  # used to parallelize sampling jobs.
  skip_batches = config.sample.get('skip_batches', 0)
  gen_data_dir = config.sample.get('gen_data_dir', None)
  is_gen = gen_data_dir is not None

  model_name = config.model.get('name')
  if not is_gen and 'upsampler' in model_name:
    logging.info('Generated low resolution not provided, using ground '
                 'truth input.')

  # Get ground truth dataset for grayscale image.
  tf_dataset = datasets.get_dataset(
      name=config.dataset,
      config=config,
      batch_size=batch_size,
      subset=subset)
  tf_dataset = tf_dataset.skip(skip_batches)
  data_iter = iter(tf_dataset)

  # Creates dataset from generated TFRecords.
  # This is used as low resolution input to the upsamplers.
  gen_iter = None
  if is_gen:
    gen_tf_dataset = datasets.get_gen_dataset(
        data_dir=gen_data_dir, batch_size=batch_size)
    gen_tf_dataset = gen_tf_dataset.skip(skip_batches)
    gen_iter = iter(gen_tf_dataset)

  store_samples(data_iter, config, logdir, subset, gen_iter)


def main(_):
  logging.info('Logging to %s.', FLAGS.logdir)
  if FLAGS.mode == 'sample_valid':
    logging.info('[main] I am the sampler.')
    sample(FLAGS.logdir, subset='valid')
  elif FLAGS.mode == 'sample_test':
    logging.info('[main] I am the sampler test.')
    sample(FLAGS.logdir, subset='test')
  elif FLAGS.mode == 'sample_train':
    logging.info('[main] I am the sampler train.')
    sample(FLAGS.logdir, subset='eval_train')
  else:
    raise ValueError(
        'Unknown mode {}. '
        'Must be one of [sample, sample_test]'.format(FLAGS.mode))


if __name__ == '__main__':
  app.run(main)
