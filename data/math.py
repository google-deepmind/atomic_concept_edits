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

"""Math dataset loading for ACE."""

import pandas as pd


def load_math_problems(
    num_prompts: int, split: str, base_path: str
) -> list[str]:
  """Loads MATH CSP problems from a CSV file.

  Args:
    num_prompts: Number of math problems to load.
    split: The dataset split (e.g. 'train', 'test').
    base_path: Path to the directory containing the CSV files.

  Returns:
    A list of math problem strings.
  """
  df = pd.read_csv(f'{base_path}/{split}.csv')
  prompts = df.sample(n=num_prompts)['Full Problem'].tolist()
  print(f'Loaded {len(prompts)} math problems.')
  return prompts
