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

"""Configuration for word count objective."""

from atomic_concept_edits import ace_exploration
from atomic_concept_edits import autoraters
from atomic_concept_edits import data
from atomic_concept_edits import models
from atomic_concept_edits import mutation_samplers
from atomic_concept_edits.util import eval as eval_util
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Returns the config for the word count objective."""
  config = config_dict.ConfigDict()

  objective = 'Force the model to adhere to a word count constraint.'
  save_path = '/tmp/ace/word_count'

  gemini_llm_config = models.GeminiModelConfig(
      model='gemini-3-flash-preview',
      api_key='[API_KEY]',
  )
  mutation_sampler_config = mutation_samplers.ACEMutationSamplerConfig(
      llm_config=gemini_llm_config,
      objective=objective,
      constitution='',
  )
  dataset_config = data.DatasetConfig(
      dataset_name='lima',
      num_prompts=2,
      split='train',
  )
  model_config = gemini_llm_config
  autorater_config = autoraters.WordCountAutoraterConfig(
      target_word_count=10,
      word_count_slope=1.0,
  )

  config.ace_exploration_config = ace_exploration.ACEExplorationConfig(
      mutation_sampler_config=mutation_sampler_config,
      target_model_config=model_config,
      autorater_config=autorater_config,
      dataset_config=dataset_config,
      sample_size_at_depth=(5, 3, 2),
      num_responses_per_prompt=1,
      min_score=0.0,
      max_score=0.0,
      score_selection_method=eval_util.AutoraterScoreSelectionMethod.ANY,
      max_parallelism_ace_generation=50,
      max_parallelism_autorater=50,
      max_parallelism_target_model=50,
      save_path=save_path,
  )

  return config
