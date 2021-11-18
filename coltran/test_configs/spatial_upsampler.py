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

"""Test configurations for color upsampler."""
from ml_collections import ConfigDict


def get_config():
  """Experiment configuration."""
  config = ConfigDict()

  # Data.
  config.dataset = 'custom'
  config.downsample = True
  config.downsample_res = 64
  config.resolution = [256, 256]
  config.random_channel = True
  config.timeline = 6
  config.mask_dir = 'D:\\Timeseries_cropped_512\\masks_final_trainset'
  config.data_dir = 'D:\\Timeseries_cropped_512\\videos_final_trainset'
  config.targets_dir = 'D:\\Timeseries_cropped_512\\targets\\spatial_upsampler'

  # Training.
  config.batch_size = 1
  config.max_train_steps = 30000  # 50000
  config.save_checkpoint_secs = 900
  config.num_epochs = -1
  config.polyak_decay = 0.999
  config.eval_num_examples = 20000
  config.eval_batch_size = 16
  config.eval_checkpoint_wait_secs = -1

  config.optimizer = ConfigDict()
  config.optimizer.type = 'rmsprop'
  config.optimizer.learning_rate = 3e-4

  # Model.
  config.model = ConfigDict()
  config.model.hidden_size = 32
  config.model.ff_size = 32
  config.model.num_heads = 4
  config.model.num_encoder_layers = 3
  config.model.resolution = [64, 64]
  config.model.name = 'spatial_upsampler'

  config.sample = ConfigDict()
  config.sample.gen_data_dir = 'D:\color_upsampler_128_12bs_100k\samples'
  config.sample.log_dir = 'samples'
  config.sample.batch_size = 1
  config.sample.mode = 'argmax'
  config.sample.num_samples = 1
  config.sample.num_outputs = 100
  config.sample.skip_batches = 0
  config.sample.gen_file = 'gen0_100_32_100k'

  return config
