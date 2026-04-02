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

"""Mutation samplers for the ACE pipeline."""

from concurrent import futures
import dataclasses
import json
import threading
from typing import Any, Callable, Union

from . import ace_set_class
from . import interface
from . import prompts


@dataclasses.dataclass
class SimpleLLMMutationSamplerConfig(interface.MutationSamplerConfig):
  """Configuration for a simple LLM mutation sampler."""

  llm_config: interface.ModelConfig
  objective: str
  instruction_preamble: str = prompts.SIMPLE_LLM_MUTATION_PREAMBLE

  def build_mutation_sampler(self) -> interface.MutationSampler:
    return SimpleLLMMutationSampler(self)


@dataclasses.dataclass
class ACEMutationSamplerConfig(interface.MutationSamplerConfig):
  """Configuration for an ACE mutation sampler."""

  llm_config: interface.ModelConfig
  objective: str
  constitution: str | None = None
  instruction_preamble: str = prompts.ACE_MUTATION_PREAMBLE

  def build_mutation_sampler(self) -> interface.MutationSampler:
    return ACEMutationSampler(self)


@dataclasses.dataclass
class ACETwoStageMutationSamplerConfig(interface.MutationSamplerConfig):
  """Configuration for a two-stage ACE mutation sampler."""

  llm_config: interface.ModelConfig
  objective: str
  concept_extraction_preamble: str
  ace_generation_preamble: str
  constitution: str | None = None

  def build_mutation_sampler(self) -> interface.MutationSampler:
    return ACETwoStageMutationSampler(self)


@dataclasses.dataclass
class ACELoopMutationSamplerConfig(ACEMutationSamplerConfig):
  """Configuration for an ACE loop mutation sampler.

  Attributes:
    expander_preamble: Preamble for the LLM prompt in the expander step.
    refiner_preamble: Preamble for the LLM prompt in the refiner step.
    is_valid_ace_fn: An optional function that validates ACEs.
    objective: The objective of the mutation task.
    constitution: An optional constitution to guide mutation.
    instruction_preamble: Preamble for LLM prompt in proposer step.
    num_correction_attempts: The number of times to attempt to correct invalid
      ACEs.
  """

  expander_preamble: str = prompts.EXPANDER_PREAMBLE
  refiner_preamble: str = prompts.REFINER_PREAMBLE
  is_valid_ace_fn: Callable[[str], bool] | None = None
  num_correction_attempts: int = 2

  def build_mutation_sampler(self) -> interface.MutationSampler:
    return ACELoopMutationSampler(self)


class SimpleLLMMutationSampler(interface.MutationSampler):
  """Simple LLM mutation sampler."""

  def __init__(self, config: SimpleLLMMutationSamplerConfig):
    self.config = config
    self.llm = config.llm_config.build_model()

  def sample(
      self, prompt: interface.Content, num_samples: int = 1
  ) -> interface.Content:
    """Mutates the prompt and returns a list of mutated prompts."""
    llm_prompt = (
        self.config.instruction_preamble
        + prompts.MUTATION_TASK_TEMPLATE.format(
            objective=self.config.objective,
            constitution=None,
            prompt=prompt,
        )
    )

    results = self.llm.generate(llm_prompt, num_responses=num_samples)
    return results


class ACEMutationSampler(interface.MutationSampler):
  """ACE mutation sampler."""

  config: ACEMutationSamplerConfig

  def __init__(self, config: ACEMutationSamplerConfig):
    self.config = config
    self.llm = config.llm_config.build_model()

  def generate_ace_set(
      self,
      prompt: interface.Content,
  ) -> ace_set_class.ACESet:
    """Generates an ACE set for the given prompt."""
    llm_prompt = (
        self.config.instruction_preamble
        + prompts.MUTATION_TASK_TEMPLATE.format(
            objective=self.config.objective,
            constitution=self.config.constitution,
            prompt=prompt,
        )
    )
    try:
      ace_set = self.llm.generate_object(llm_prompt, ace_set_class.ACESet)
    except Exception as e:  # pylint: disable=broad-except
      print(f'Error generating ACE set: {e}')
      ace_set = ace_set_class.ACESet(aces=[], prompt=prompt)
    return ace_set

  def sample_aces(
      self, prompt: interface.Content, num_samples: int = 1
  ) -> list[ace_set_class.ACE]:
    """Samples ACEs for the given prompt."""
    ace_set = self.generate_ace_set(prompt)
    all_aces = ace_set.aces
    top_aces = sorted(all_aces, key=lambda x: x.ace_score, reverse=True)[
        :num_samples
    ]
    return top_aces

  def sample(
      self, prompt: interface.Content, num_samples: int = 1
  ) -> interface.Content:
    """Mutates the prompt and returns a list of mutated prompts."""
    aces = self.sample_aces(prompt, num_samples)
    result = []
    for ace in aces:
      result.append(ace.updated_prompt)
    return result


class ACETwoStageMutationSampler(ACEMutationSampler):
  """ACE Two Stage mutation sampler."""

  def __init__(self, config: ACETwoStageMutationSamplerConfig):
    super().__init__(
        ACEMutationSamplerConfig(
            llm_config=config.llm_config,
            objective=config.objective,
            constitution=config.constitution,
            instruction_preamble='',
        )
    )
    self.concept_extraction_preamble = config.concept_extraction_preamble
    self.ace_generation_preamble = config.ace_generation_preamble

  def sample_aces(
      self, prompt: interface.Content, num_samples: int = 1
  ) -> list[ace_set_class.ACE]:
    """Samples ACEs for the given prompt."""
    llm_prompt = (
        self.concept_extraction_preamble
        + prompts.MUTATION_TASK_TEMPLATE.format(
            objective=self.config.objective,
            constitution=self.config.constitution,
            prompt=prompt,
        )
    )
    concept_set = self.llm.generate_object(llm_prompt, ace_set_class.ConceptSet)

    print(f'    {len(concept_set.concepts)} concepts generated.')

    results = []
    with futures.ThreadPoolExecutor(
        max_workers=len(concept_set.concepts)
    ) as executor:
      for concept in concept_set.concepts:
        llm_prompt = (
            self.ace_generation_preamble
            + prompts.ACE_GENERATION_TASK_TEMPLATE.format(
                objective=self.config.objective,
                constitution=self.config.constitution,
                prompt=prompt,
                concept=concept.to_json(),
            )
        )
        results.append(
            executor.submit(
                self.llm.generate_object,
                llm_prompt,
                ace_set_class.ACESetForConcept,
            )
        )
      executor.shutdown(wait=True)

    all_aces = []
    for result in results:
      all_aces.extend(result.result().aces)

    top_aces = sorted(all_aces, key=lambda x: x.ace_score, reverse=True)[
        :num_samples
    ]
    return top_aces


class ACELoopMutationSampler(interface.MutationSampler):
  """ACE loop mutation sampler with proposer, expander, and refiner steps."""

  config: ACELoopMutationSamplerConfig

  def __init__(self, config: ACELoopMutationSamplerConfig):
    self.config = config
    self.llm = config.llm_config.build_model()

  def _try_correcting_invalid_aces(
      self,
      ace_set: ace_set_class.ACESet,
      num_samples: int,
      invalid_aces_log: Union[list[dict[str, Any]], None] = None,
      lock: Union[threading.Lock, None] = None,
  ):
    """Filters and corrects aces in an ACESet based on `is_valid_ace_fn`."""
    if self.config.is_valid_ace_fn is None:
      return

    aces_to_correct = []
    valid_aces = []

    with futures.ThreadPoolExecutor() as executor:
      future_to_ace = {
          executor.submit(self.config.is_valid_ace_fn, ace.updated_prompt): ace
          for ace in ace_set.aces
      }
      for future in futures.as_completed(future_to_ace):
        ace = future_to_ace[future]
        error = None
        try:
          if future.result():
            valid_aces.append(ace)
          else:
            error = 'Invalid ACE based on is_valid_ace_fn'
        except (ValueError, TypeError) as e:
          error = str(e)

        if error:
          ace_info = {
              'ace_type': ace.ace_type.name,
              'verbalization': ace.verbalization,
              'updated_prompt': ace.updated_prompt,
              'error': error,
              'associated_concept': (
                  ace.associated_concept.to_json()
                  if ace.associated_concept
                  else None
              ),
          }
          aces_to_correct.append((ace, ace_info))
    ace_set.aces = valid_aces

    for attempt in range(self.config.num_correction_attempts):
      if not aces_to_correct or len(ace_set.aces) >= num_samples:
        break

      print(
          f'  Attempting to correct {len(aces_to_correct)} aces (attempt'
          f' {attempt + 1}/{self.config.num_correction_attempts}).'
      )
      corrector_prompt = self.config.refiner_preamble.format(
          objective=self.config.objective,
          aces_to_correct=json.dumps(
              [info for _, info in aces_to_correct], indent=2
          ),
      )
      aces_to_correct_next_attempt = []
      try:
        corrected_aces_json = self.llm.generate_object(
            corrector_prompt, ace_set_class.ACESet
        )
        corrected_aces = {a.verbalization: a for a in corrected_aces_json.aces}
        print(f'  LLM returned {len(corrected_aces)} corrected aces.')

        corrected_aces_to_validate = []
        for _, info in aces_to_correct:
          if info['verbalization'] in corrected_aces:
            corrected_aces_to_validate.append(
                (corrected_aces[info['verbalization']], info)
            )
          elif invalid_aces_log is not None:
            if lock:
              with lock:
                invalid_aces_log.append(info)
            else:
              invalid_aces_log.append(info)

        with futures.ThreadPoolExecutor() as executor:
          future_to_ace_info = {
              executor.submit(self.config.is_valid_ace_fn, ca.updated_prompt): (
                  ca,
                  info,
              )
              for ca, info in corrected_aces_to_validate
          }

          for future in futures.as_completed(future_to_ace_info):
            corrected_ace, info = future_to_ace_info[future]
            error = None
            try:
              if future.result():
                ace_set.aces.append(corrected_ace)
              else:
                error = 'Corrected ACE is still invalid'
            except (ValueError, TypeError) as e:
              error = f'Validation of corrected ACE failed: {e}'

            if error:
              info['updated_prompt'] = corrected_ace.updated_prompt
              info['error'] = error
              aces_to_correct_next_attempt.append((corrected_ace, info))
        aces_to_correct = aces_to_correct_next_attempt
      except (json.JSONDecodeError, TypeError) as e:
        print(f'  Error during ace correction: {e}')
        if invalid_aces_log is not None:
          for _, info in aces_to_correct:
            if lock:
              with lock:
                invalid_aces_log.append(info)
            else:
              invalid_aces_log.append(info)
        aces_to_correct = []

    if aces_to_correct and invalid_aces_log is not None:
      for _, info in aces_to_correct:
        if lock:
          with lock:
            invalid_aces_log.append(info)
        else:
          invalid_aces_log.append(info)

  def sample_aces(
      self, prompt: interface.Content, num_samples: int = 1
  ) -> list[ace_set_class.ACE]:
    """Samples ACEs for the given prompt."""
    invalid_aces: list[dict[str, Any]] = []
    llm_prompt = (
        self.config.instruction_preamble
        + prompts.MUTATION_TASK_TEMPLATE.format(
            objective=self.config.objective,
            constitution=self.config.constitution,
            prompt=prompt,
        )
    )
    try:
      initial_ace_set = self.llm.generate_object(
          llm_prompt, ace_set_class.ACESet
      )
    except json.JSONDecodeError:
      print(
          'JSON decode error on initial ace set generation for prompt:'
          f' {prompt}'
      )
      initial_ace_set = ace_set_class.ACESet(prompt=prompt, aces=[])
    self._try_correcting_invalid_aces(
        initial_ace_set, num_samples, invalid_aces
    )

    if len(initial_ace_set.aces) < num_samples:
      print(f'  {len(initial_ace_set.aces)} ACEs generated. Expanding...')
      existing_aceset_json = json.dumps(initial_ace_set.to_json(), indent=2)
      expander_prompt = self.config.expander_preamble.format(
          objective=self.config.objective,
          constitution=self.config.constitution,
          existing_aceset=existing_aceset_json,
      )
      try:
        expanded_ace_set = self.llm.generate_object(
            expander_prompt, ace_set_class.ACESet
        )
      except json.JSONDecodeError:
        print(f'JSON decode error on expanded ace set for prompt: {prompt}')
        expanded_ace_set = initial_ace_set
      self._try_correcting_invalid_aces(
          expanded_ace_set, num_samples, invalid_aces
      )
      all_aces = expanded_ace_set.aces
    else:
      all_aces = initial_ace_set.aces

    top_aces = sorted(all_aces, key=lambda x: x.ace_score, reverse=True)[
        :num_samples
    ]
    return top_aces

  def sample(
      self, prompt: interface.Content, num_samples: int = 1
  ) -> interface.Content:
    """Mutates the prompt and returns a list of mutated prompts."""
    aces = self.sample_aces(prompt, num_samples)
    result = []
    for ace in aces:
      result.append(ace.updated_prompt)
    return result
