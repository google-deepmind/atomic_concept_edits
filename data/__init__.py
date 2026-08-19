# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data loading for ACE."""

from . import config
from . import custom
from . import math

# Optional dataset loaders are lazily imported to avoid requiring
# optional dependencies.
try:
  from . import huggingface  # pylint: disable=g-import-not-at-top
except ImportError:
  huggingface = None

try:
  from . import tfds  # pylint: disable=g-import-not-at-top
except ImportError:
  tfds = None

DatasetConfig = config.DatasetConfig


class DatasetFactory:
  """Factory for datasets."""

  def __call__(self, dataset_config: config.DatasetConfig) -> list[str]:
    """Returns a list of prompts based on the config."""
    if dataset_config.dataset_name == 'coco_captions':
      if tfds is None:
        raise ImportError(
            'tensorflow_datasets is required for coco_captions. '
            'Install with: pip install -e ".[tfds]"'
        )
      return tfds.load_coco_captions(
          dataset_config.num_prompts, dataset_config.split
      )
    elif dataset_config.dataset_name == 'lima':
      if huggingface is None:
        raise ImportError(
            'datasets is required for lima. '
            'Install with: pip install -e ".[hf]"'
        )
      return huggingface.load_lima_prompts(
          dataset_config.num_prompts, dataset_config.split
      )
    elif dataset_config.prompt_list is not None:
      return custom.load_custom_dataset(
          dataset_config.prompt_list, dataset_config.num_prompts
      )
    elif dataset_config.dataset_name == 'math_problems':
      if dataset_config.data_path is None:
        raise ValueError(
            'data_path must be provided for math_problems dataset.'
        )
      return math.load_math_problems(
          dataset_config.num_prompts,
          dataset_config.split,
          dataset_config.data_path,
      )
    else:
      raise ValueError(f'Unsupported dataset config: {dataset_config}')
