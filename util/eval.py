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

"""Evaluation utility functions for ACE generation."""

import dataclasses
import enum
import os
from typing import Any, Sequence

import pandas as pd

from .. import interface
from . import parallel_util

DISCOUNT_FACTOR = 0.99


class AutoraterScoreSelectionMethod(enum.Enum):
  """Method to use for checking the autorater scores."""

  ANY = 'any'
  ALL = 'all'


def get_current_decay_value(
    initial_value: float, final_value: float, current_epoch: int, epochs: int
):
  """Returns the current decay value to use.

  Args:
    initial_value: The initial value to use.
    final_value: The final value to use.
    current_epoch: The current epoch of the optimization.
    epochs: The number of epochs to optimize the constitution for.

  Returns:
    The current value to use.
  """
  if epochs > 1:
    current_value = int(
        initial_value
        + (final_value - initial_value) * current_epoch / (epochs - 1)
    )
  else:
    current_value = final_value
  if initial_value <= final_value:
    return max(initial_value, min(final_value, current_value))
  else:
    return min(initial_value, max(final_value, current_value))


def is_objective_satisfied(
    autorater_scores: list[float],
    min_score: float,
    max_score: float,
    selection_method: AutoraterScoreSelectionMethod = AutoraterScoreSelectionMethod.ANY,
) -> bool:
  """Checks if the objective is satisfied by the autorater scores.

  Args:
    autorater_scores: The list of autorater scores.
    min_score: The minimum score.
    max_score: The maximum score.
    selection_method: The selection method to use for checking the scores.

  Returns:
    True if the objective is satisfied, False otherwise.
  """
  if not autorater_scores:
    return False
  if selection_method == AutoraterScoreSelectionMethod.ANY:
    for score in autorater_scores:
      if score >= min_score and score <= max_score:
        return True
    return False
  else:
    for score in autorater_scores:
      if score < min_score or score > max_score:
        return False
    return True


def generate_target_model_response(
    target_model: interface.Model,
    prompt: str,
    num_responses: int = 1,
    save_path_for_responses: str | None = None,
) -> interface.Content | None:
  """Generates responses from the target model and saves them."""
  responses = None
  try:
    responses = target_model.generate(
        prompt=prompt,
        num_responses=num_responses,
        save_path=save_path_for_responses,
    )
  except Exception as e:  # pylint: disable=broad-except
    print(f'Error generating response for prompt: {prompt}, {e}')
  return responses


def generate_autorater_scores(
    autorater: interface.Autorater,
    prompt: str,
    responses: interface.Content | None = None,
    response_paths: list[str] | None = None,
    num_responses_per_prompt: int = 1,
    root_prompt: str | None = None,
) -> list[float]:
  """Scores responses using the autorater."""
  if responses is None and response_paths is None:
    raise ValueError('Either responses or response_paths must be provided.')

  kwargs_list = []
  for i in range(num_responses_per_prompt):
    if num_responses_per_prompt > 1 and responses is not None:
      response = responses[i]
    else:
      response = responses
    response_path = None if response_paths is None else response_paths[i]
    kwargs_list.append({
        'prompt': prompt,
        'response': response,
        'response_path': response_path,
        'root_prompt': root_prompt,
    })
  autorater_scores = parallel_util.run_in_parallel(
      function=autorater.score,
      list_of_kwargs=kwargs_list,
      num_workers=min(interface.MAX_PARALLELISM, len(kwargs_list)),
  )
  return autorater_scores


def evaluate_prompt_and_save_response(
    target_model: interface.Model,
    autorater: interface.Autorater | None,
    prompt: str,
    num_responses: int,
    save_path: str,
) -> tuple[list[float] | None, list[Any] | None]:
  """Evaluates a prompt and saves the response."""

  responses = generate_target_model_response(
      target_model, prompt, num_responses, save_path
  )
  if autorater is not None and responses is not None:
    return generate_autorater_scores(autorater, prompt, responses), responses
  return None, responses


def score_all_nodes(
    exploration_data: pd.DataFrame,
    objective_satisfied_column: str = 'objective_satisfied',
) -> pd.DataFrame:
  """Scores all nodes in the exploration tree.

  Args:
    exploration_data: The exploration data.
    objective_satisfied_column: The column to use for checking the objective
      satisfied.

  Returns:
    The exploration data with the objective satisfied score column.
  """
  max_depth = max(exploration_data['depth']) + 1
  objective_satisfied_score_column = f'{objective_satisfied_column}_score'
  exploration_data[objective_satisfied_score_column] = None
  for depth in range(max_depth - 1, -1, -1):
    print(f'Scoring all nodes at depth: {depth}')
    for _, row in exploration_data[
        exploration_data['depth'] == depth
    ].iterrows():
      children_scores = exploration_data[
          (exploration_data['parent_id'] == row['prompt_id'])
          & (pd.notna(exploration_data[objective_satisfied_column]))
      ][objective_satisfied_score_column].values
      if children_scores.size > 0:
        objective_satisfied_score = (
            DISCOUNT_FACTOR * sum(children_scores) / len(children_scores)
        )
      else:
        if pd.isna(row[objective_satisfied_column]):
          objective_satisfied_score = 1
        else:
          objective_satisfied_score = int(row[objective_satisfied_column])
      exploration_data.loc[
          exploration_data['prompt_id'] == row['prompt_id'],
          objective_satisfied_score_column,
      ] = objective_satisfied_score
  return exploration_data


@dataclasses.dataclass
class Metrics:
  total_leaves: int = 0
  successful_leaves: int = 0
  success_rate_among_leaves: float = 0.0
  root_prompts: int = 0
  success_rate_among_root_prompts: float = 0.0
  success_rate_among_unsuccessful_root_prompts: float = 0.0
  success_rate_at_depth: dict[int, float] = dataclasses.field(
      default_factory=dict
  )


def compute_metrics(
    exploration_data: pd.DataFrame,
    sample_size_at_depth: Sequence[int],
    objective_satisfied_column: str = 'objective_satisfied',
    save_metrics_path: str | None = None,
) -> Metrics:
  """Computes the success rate of ACEs and BLEU scores.

  Args:
    exploration_data: The exploration data to compute the success rate for.
    sample_size_at_depth: The number of actions to sample at each depth.
    objective_satisfied_column: The column to use for checking the objective
    save_metrics_path: The path to directory to save the metrics.csv file.

  Returns:
    A Metrics dataclass with all computed metrics.
  """

  all_leaves = exploration_data[
      exploration_data[objective_satisfied_column].notna()
  ]
  print(f'Total leaves: {len(all_leaves)}')

  successful_leaves = all_leaves[all_leaves[objective_satisfied_column] == True]  # pylint: disable=singleton-comparison
  print(f'Number of successful leaves: {len(successful_leaves)}')
  print(
      f'Success rate among leaves: {len(successful_leaves) / len(all_leaves)}'
  )

  root_prompts = exploration_data[exploration_data['depth'] == 0]['root_id']
  unsuccessful_root_prompts = exploration_data[
      (exploration_data['depth'] == 0)
      & (exploration_data[objective_satisfied_column] == False)  # pylint: disable=singleton-comparison
  ]['root_id']

  led_to_success = exploration_data[
      exploration_data['root_id'].isin(root_prompts)
      & (exploration_data[objective_satisfied_column] == True)  # pylint: disable=singleton-comparison
  ]
  led_to_success_from_unsuccessful = exploration_data[
      exploration_data['root_id'].isin(unsuccessful_root_prompts)
      & (exploration_data[objective_satisfied_column] == True)  # pylint: disable=singleton-comparison
  ]

  ace_success_rate = len(led_to_success.root_id.unique()) / len(root_prompts)
  ace_unsuccessful_success_rate = len(
      led_to_success_from_unsuccessful.root_id.unique()
  ) / len(unsuccessful_root_prompts)

  print(f'Success rate of ACE among root prompts: {ace_success_rate}')
  print(
      'Success rate of ACE among root prompts that were initially'
      f' unsuccessful: {ace_unsuccessful_success_rate}'
  )

  depth_success_rates = {}
  for depth in range(len(sample_size_at_depth)):
    successful_at_depth = led_to_success[
        led_to_success['depth'] == depth
    ].root_id.unique()
    rate = len(successful_at_depth) / len(root_prompts)
    depth_success_rates[depth] = rate
    print(f'Success rate of ACE at depth {depth}: {rate}')

  metrics = Metrics(
      total_leaves=len(all_leaves),
      successful_leaves=len(successful_leaves),
      success_rate_among_leaves=len(successful_leaves) / len(all_leaves),
      root_prompts=len(root_prompts),
      success_rate_among_root_prompts=ace_success_rate,
      success_rate_among_unsuccessful_root_prompts=ace_unsuccessful_success_rate,
      success_rate_at_depth=depth_success_rates,
  )

  if save_metrics_path is not None:
    metrics_df = pd.DataFrame([dataclasses.asdict(metrics)])
    metrics_df.to_csv(
        os.path.join(save_metrics_path, 'metrics.csv'), index=False
    )

  return metrics
