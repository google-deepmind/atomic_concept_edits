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

"""Hugging Face dataset loading for ACE."""

import datasets


def load_lima_prompts(num_prompts: int, split: str) -> list[str]:
  """Loads LIMA prompts from the Hugging Face dataset."""
  ds = datasets.load_dataset('GAIR/lima', split=split)
  if not isinstance(ds, datasets.Dataset):
    raise TypeError(f'Expected a Dataset, got {type(ds)}')
  prompts_list = []
  for example in ds.select(range(min(num_prompts, len(ds)))):
    prompt = example['conversations'][0].strip()
    prompts_list.append(prompt)
  print(f'Loaded {len(prompts_list)} prompts.')
  return prompts_list
