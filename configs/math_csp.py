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

"""Configuration for math CSP objective."""

from atomic_concept_edits import ace_exploration
from atomic_concept_edits import autoraters
from atomic_concept_edits import data
from atomic_concept_edits import models
from atomic_concept_edits import mutation_samplers
from atomic_concept_edits.util import eval as eval_util
from ml_collections import config_dict


def is_valid_math_csp(prompt: str) -> bool:
  """Checks if a prompt is a valid and solvable CSP."""
  try:
    equations, target_variable = autoraters.parse_csp(prompt)
    autoraters.solve_csp(equations, target_variable)
    return True
  except Exception:  # pylint: disable=broad-except
    return False


def get_config() -> config_dict.ConfigDict:
  """Returns the config for the math CSP objective."""
  config = config_dict.ConfigDict()

  objective = (
      'Modify the math problem to make it more difficult to solve correctly'
      ' while ensuring it remains a valid algebra problem.'
  )
  save_path = '/tmp/ace/math_csp'

  gemini_llm_config = models.GeminiModelConfig(
      model='gemini-3-flash-preview',
      api_key='[API_KEY]',
  )

  mutation_sampler_config = mutation_samplers.ACELoopMutationSamplerConfig(
      llm_config=gemini_llm_config,
      objective=objective,
      constitution='',
      is_valid_ace_fn=is_valid_math_csp,
  )

  dataset_config = data.DatasetConfig(
      dataset_name='math_problems',
      num_prompts=10,
      split='train',
      data_path='[DATA_PATH]',
  )

  solver_config = gemini_llm_config
  csp_autorater_config = autoraters.CSPAutoraterConfig(
      llm_config=gemini_llm_config,
  )

  config.ace_exploration_config = ace_exploration.ACEExplorationConfig(
      mutation_sampler_config=mutation_sampler_config,
      target_model_config=solver_config,
      autorater_config=csp_autorater_config,
      dataset_config=dataset_config,
      sample_size_at_depth=(2, 2, 2),
      num_responses_per_prompt=3,
      min_score=1,
      max_score=1,
      score_selection_method=eval_util.AutoraterScoreSelectionMethod.ANY,
      max_parallelism_ace_generation=50,
      max_parallelism_autorater=50,
      max_parallelism_target_model=50,
      save_path=save_path,
  )

  return config
