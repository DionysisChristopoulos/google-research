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
from os import getenv

resolution = int(getenv("DOWNSAMPLE_SIZE", 64))
model_size = int(getenv("MODEL_SIZE", 128))

def get_config():
  """Experiment configuration."""
  config = ConfigDict()

  # Data.
  config.dataset = 'custom'
  config.downsample = True
  config.downsample_res = resolution
  config.resolution = [256, 256]
  config.random_channel = True
  config.timeline = 6
  config.mask_dir = './Datasets/Timeseries_cropped_512/masks_final_trainset'
  config.data_dir = './Datasets/Timeseries_cropped_512/videos_final_trainset'
  config.targets_dir = './Datasets/inpaint_new_sr'
  config.mask_availability = True


  # Training.
  config.batch_size = 1
  config.max_train_steps = 15000
  config.save_checkpoint_secs = 900
  config.num_epochs = -1
  config.polyak_decay = 0.999
  config.eval_num_examples = 20000
  config.eval_batch_size = 1
  config.eval_checkpoint_wait_secs = -1

  config.optimizer = ConfigDict()
  config.optimizer.type = 'rmsprop'
  config.optimizer.learning_rate = 3e-4

  print("Model size: {}".format(model_size))
  # Model.
  config.model = ConfigDict()
  config.model.hidden_size = model_size
  config.model.ff_size = model_size
  config.model.num_heads = 4
  config.model.num_encoder_layers = 3
  config.model.resolution = [resolution, resolution]
  config.model.name = 'spatial_upsampler'

  return config
