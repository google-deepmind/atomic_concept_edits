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

"""Interface for target models, autoraters and tasks."""

import abc
import dataclasses
import pydantic
from . import data_types

Content = data_types.Content


MAX_PARALLELISM = 100

TEXT_EXTENSIONS = ('.txt', '.json')

PROMPT_ID_COLUMN = 'prompt_id'
DEPTH_COLUMN = 'depth'
PROMPT_COLUMN = 'prompt'
PARENT_ID_COLUMN = 'parent_id'
PARENT_PROMPT_COLUMN = 'parent_prompt'
ACE_VERBALIZATION_COLUMN = 'ace_verbalization'
ACE_SCORE_COLUMN = 'ace_score'
ACE_CONSTITUTION_STRATEGY_COLUMN = 'ace_constitution_strategy'
ROOT_PROMPT_COLUMN = 'root_prompt'
ROOT_ID_COLUMN = 'root_id'
TARGET_MODEL_RESPONSES_COLUMN = 'target_model_responses'
TARGET_MODEL_RESPONSE_PATHS_COLUMN = 'target_model_response_paths'
AUTORATER_SCORES_COLUMN = 'autorater_scores'
OBJECTIVE_SATISFIED_COLUMN = 'objective_satisfied'


class Model(abc.ABC):
  """Interface for target models."""

  @abc.abstractmethod
  def generate(
      self,
      prompt: Content,
      load_response: bool = True,
      num_responses: int = 1,
      save_path: str | None = None,
      include_thoughts: bool = False,
  ) -> Content:
    """Generates response from the target model."""
    raise NotImplementedError()

  @abc.abstractmethod
  def generate_object(
      self, prompt: Content, cls: pydantic.BaseModel
  ) -> pydantic.BaseModel:
    """Generates a structured object from the model."""
    raise NotImplementedError()


@dataclasses.dataclass
class ModelConfig(abc.ABC):
  """Configuration for a model."""

  @abc.abstractmethod
  def build_model(self) -> Model:
    """Converts the config to a model."""
    raise NotImplementedError()


class Autorater(abc.ABC):
  """Interface for autoraters."""

  @abc.abstractmethod
  def score(
      self,
      prompt: Content | None,
      response: Content,
      response_path: str | None = None,
      root_prompt: Content | None = None,
  ) -> float:
    """Scores an example."""
    raise NotImplementedError()


@dataclasses.dataclass
class AutoraterConfig(abc.ABC):
  """Configuration for an autorater."""

  @abc.abstractmethod
  def build_autorater(self) -> Autorater:
    """Converts the config to an autorater."""
    raise NotImplementedError()


class Scorer(abc.ABC):
  """General purpose Scorer.

  A scorer directly scores the prompt. May use a model
  and autorater internally to generate and score responses.
  """

  @abc.abstractmethod
  def generate_and_score(
      self,
      prompt: Content,
      num_responses: int = 1,
      save_path: str | None = None,
      root_prompt: Content | None = None,
  ) -> tuple[Content, list[float]]:
    """Generates responses and scores them.

    Args:
      prompt: The prompt to score.
      num_responses: The number of responses to generate.
      save_path: Optional path to save the responses to.
      root_prompt: Optional root prompt for autorater context.

    Returns:
      A tuple of (responses, scores) where responses is the model output and
      scores is a list of autorater scores, one per response.
    """
    raise NotImplementedError()


@dataclasses.dataclass
class ScorerConfig(abc.ABC):
  """Configuration for a scorer."""

  @abc.abstractmethod
  def build_scorer(self) -> Scorer:
    """Builds the scorer from the config."""
    raise NotImplementedError()


class MutationSampler(abc.ABC):
  """Interface for mutation samplers."""

  @abc.abstractmethod
  def sample(self, prompt: Content, num_samples: int = 1) -> Content:
    """Mutates the prompt and returns a list of mutated prompts."""
    raise NotImplementedError()


@dataclasses.dataclass
class MutationSamplerConfig(abc.ABC):
  """Configuration for a mutation sampler."""

  @abc.abstractmethod
  def build_mutation_sampler(self) -> MutationSampler:
    """Converts the config to a mutation sampler."""
    raise NotImplementedError()
