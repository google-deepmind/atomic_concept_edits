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

r"""Run constitution optimizer.

Example usage:
python run_constitution_optimizer.py -- \
--exploration_output_dir='<path_to_exploration_data.csv directory>'
"""

from collections.abc import Sequence
from absl import app
from absl import flags
from atomic_concept_edits import constitution_optimizer
from atomic_concept_edits import models


_EXPLORATION_OUTPUT_DIR = flags.DEFINE_string(
    'exploration_output_dir',
    None,
    'Path to saved exploration results directory. It must be of the form'
    ' <save_path>/<run_id> and exploration_data.csv MUST exist in this'
    ' directory.',
)


def get_config(
    exploration_output_dir: str,
) -> constitution_optimizer.ConstitutionOptimizerConfig:
  save_path = '/'.join(exploration_output_dir.split('/')[:-1])
  run_id = exploration_output_dir.split('/')[-1]
  return constitution_optimizer.ConstitutionOptimizerConfig(
      run_id=run_id,
      objective='Force the model to adhere to a word count constraint.',
      engine_llm_config=models.GeminiModelConfig(
          model='gemini-3-flash-preview',
          api_key='[API_KEY]',
      ),
      save_path=save_path,
      epochs=10,
      batch_size=50,
      objective_satisfied_column='objective_satisfied',
      objective_satisfied_score_column='objective_satisfied_score',
      initial_num_strategies=5,
      final_num_strategies=10,
      initial_change_percentage=100,
      final_change_percentage=10,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  optimizer_config = get_config(_EXPLORATION_OUTPUT_DIR.value)
  optimizer = constitution_optimizer.ConstitutionOptimizer(optimizer_config)
  optimizer.run_optimizer()


if __name__ == '__main__':
  app.run(main)
