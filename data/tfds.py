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

"""TFDS dataset loading for ACE."""

import tensorflow_datasets as tfds


def load_coco_captions(num_prompts: int, split: str) -> list[str]:
  """Loads COCO captions from the TFDS dataset."""
  ds = tfds.load('coco_captions', split=split)
  captions_list = []
  for example in ds.take(num_prompts):
    first_caption = (
        example['captions']['text'][0].numpy().decode('utf-8').strip()
    )
    captions_list.append(first_caption)
  print(f'Loaded {len(captions_list)} captions.')
  return captions_list
