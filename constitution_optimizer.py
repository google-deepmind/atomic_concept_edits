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

"""Constitution optimizer for ACE."""

import dataclasses
import json
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import pydantic

from . import interface
from . import prompts
from .util import eval as eval_util
from .util import io as io_util
from .util import parallel_util


MAX_LOSS = 10000


@dataclasses.dataclass
class ConstitutionOptimizerConfig:
  """Configuration for the constitution optimization.

  save_path: The path to save the constitution and results to.
  run_id: The run ID of the exploration tree.
  epochs: The number of epochs to optimize the constitution for.
  engine_llm: The LLM engine to use for generation.
  batch_size: The number of examples to use in each batch.
  objective: The objective of the exploration.
  objective_satisfied_column: The column name for a binary score (or NA)
    indicating if the objective is satisfied.
  objective_satisfied_score_column: The column name for a numerical score for
    how satisfied the objective is. 0 means objective is not satisfied.
  initial_num_strategies: The initial number of strategies to use.
  final_num_strategies: The final number of strategies to use.
  initial_change_percentage: The initial change percentage to use.
  final_change_percentage: The final change percentage to use.
  """

  run_id: str
  objective: str
  engine_llm_config: interface.ModelConfig
  save_path: str = '/tmp/ace/'
  epochs: int = 10
  batch_size: int = 10
  objective_satisfied_column: str = 'objective_satisfied'
  objective_satisfied_score_column: str = 'objective_satisfied_score'
  initial_constitution: str | None = None
  initial_num_strategies: int = 5
  final_num_strategies: int = 10
  initial_change_percentage: float = 100
  final_change_percentage: float = 10


class FloatResponse(pydantic.BaseModel):
  value: float


class ConstitutionOptimizer:
  """Constitution optimizer for ACE."""

  def __init__(self, config: ConstitutionOptimizerConfig):
    """Initializes the constitution optimizer."""
    self.config = config
    self.engine_llm = config.engine_llm_config.build_model()
    constitution_run_dir = (
        f'label_column={config.objective_satisfied_column},'
        f'constitution_epochs={config.epochs},'
        f'batch_size={config.batch_size},'
        f'initial_num_strategies={config.initial_num_strategies},'
        f'final_num_strategies={config.final_num_strategies},'
        f'initial_change_percentage={config.initial_change_percentage},'
        f'final_change_percentage={config.final_change_percentage}'
    )
    constitution_save_path = os.path.join(
        config.save_path, config.run_id, 'constitution', constitution_run_dir
    )

    os.makedirs(constitution_save_path, exist_ok=True)
    self.constitution_save_path = constitution_save_path

    self.train_data, self.val_data, self.test_data = (
        self.get_train_test_val_data()
    )

  def get_train_test_val_data(
      self,
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits the exploration data into train, test and val sets."""
    if os.path.exists(
        os.path.join(self.constitution_save_path, 'train_data.json')
    ):
      print(
          'Loading saved train, val and test data from'
          f' {self.constitution_save_path}...'
      )
      with open(
          os.path.join(self.constitution_save_path, 'train_data.json'), 'r'
      ) as f:
        train_data = json.load(f)
      with open(
          os.path.join(self.constitution_save_path, 'val_data.json'), 'r'
      ) as f:
        val_data = json.load(f)
      with open(
          os.path.join(self.constitution_save_path, 'test_data.json'), 'r'
      ) as f:
        test_data = json.load(f)
      return train_data, val_data, test_data

    exploration_data_path = os.path.join(
        self.config.save_path, self.config.run_id, 'exploration_data.csv'
    )
    if not os.path.exists(exploration_data_path):
      raise FileNotFoundError(
          f'Exploration data csv not found at {exploration_data_path}'
      )
    print(f'Loading exploration data from {exploration_data_path}...')
    exploration_data = pd.read_csv(exploration_data_path)
    labeled_exploration_data = exploration_data[
        exploration_data[self.config.objective_satisfied_column].notna()
        & exploration_data['parent_id'].notna()
        & exploration_data['ace_verbalization'].notna()
    ]
    reward_dataset = []
    for _, row in labeled_exploration_data.iterrows():
      satisfied_score = row[self.config.objective_satisfied_score_column]
      if 'parent_prompt' not in row:
        parent_row = exploration_data[
            exploration_data['prompt_id'] == row['parent_id']
        ]
        parent_prompt = parent_row['prompt'].values[0]
      else:
        parent_prompt = row['parent_prompt']
      reward_dataset.append({
          'prompt': parent_prompt,
          'ace_verbalization': row['ace_verbalization'],
          'satisfied_score': round(satisfied_score, 2),
          'example_desc_for_llm': (
              f"""Prompt: {parent_prompt}
Action: {row['ace_verbalization']}
"""
          ),
      })

    print(f'Number of examples in the dataset: {len(reward_dataset)}')

    objective_satisfied_data = [
        example for example in reward_dataset if example['satisfied_score'] == 1
    ]
    objective_partially_satisfied_data = [
        example
        for example in reward_dataset
        if example['satisfied_score'] > 0 and example['satisfied_score'] < 1
    ]
    objective_unsatisfied_data = [
        example for example in reward_dataset if example['satisfied_score'] == 0
    ]
    select_count = min(
        len(objective_satisfied_data),
        len(objective_unsatisfied_data)
        + len(objective_partially_satisfied_data),
    )
    reward_dataset = (
        random.sample(objective_satisfied_data, select_count)
        + random.sample(
            objective_partially_satisfied_data,
            min(len(objective_partially_satisfied_data), select_count // 2),
        )
        + random.sample(objective_unsatisfied_data, select_count // 2)
    )
    print(
        f'Balanced dataset with {select_count} examples of satisfied, '
        f'{min(len(objective_partially_satisfied_data), select_count // 2)} '
        'examples of partially satisfied and '
        f'{select_count // 2} examples of unsatisfied. '
        f'Total examples: {len(reward_dataset)}'
    )

    random.shuffle(reward_dataset)
    train_data = reward_dataset[: int(len(reward_dataset) * 0.8)]
    val_data = reward_dataset[
        int(len(reward_dataset) * 0.8) : int(len(reward_dataset) * 0.9)
    ]
    test_data = reward_dataset[int(len(reward_dataset) * 0.9) :]
    with open(
        os.path.join(self.constitution_save_path, 'train_data.json'), 'w'
    ) as f:
      json.dump(train_data, f)
    with open(
        os.path.join(self.constitution_save_path, 'val_data.json'), 'w'
    ) as f:
      json.dump(val_data, f)
    with open(
        os.path.join(self.constitution_save_path, 'test_data.json'), 'w'
    ) as f:
      json.dump(test_data, f)
    print(f'Saved train, val and test data to {self.constitution_save_path}...')
    return train_data, val_data, test_data

  def save_checkpoint(
      self,
      latest_constitution: str,
      best_constitution_on_test_set: str,
      current_epoch: int,
      train_data_with_results: list[dict[str, Any]],
      val_data_with_results: list[dict[str, Any]],
      test_data_with_results: list[dict[str, Any]],
      all_train_losses: list[float],
      all_val_losses: list[float],
      all_test_losses: list[float],
  ):
    """Saves the checkpoint."""
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/constitution.json',
        {
            'best_constitution_on_test_set': best_constitution_on_test_set,
            'latest_constitution': latest_constitution,
            'epoch': current_epoch,
        },
    )
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/train_data_with_results.json',
        train_data_with_results,
    )
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/val_data_with_results.json',
        val_data_with_results,
    )
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/test_data_with_results.json',
        test_data_with_results,
    )
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/train_losses.json', all_train_losses
    )
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/val_losses.json', all_val_losses
    )
    io_util.write_json_to_file(
        f'{self.constitution_save_path}/test_losses.json', all_test_losses
    )

  def load_checkpoint(
      self,
  ) -> tuple[Any, ...]:
    """Loads the checkpoint if it exists."""
    train_data_with_results = self.train_data
    val_data_with_results = self.val_data
    test_data_with_results = self.test_data
    all_train_losses = []
    all_val_losses = []
    all_test_losses = []
    latest_constitution = None
    best_constitution_on_test_set = None
    current_epoch = 0
    constitution_path = f'{self.constitution_save_path}/constitution.json'
    if os.path.exists(constitution_path):
      print(f'Loading saved constitution from {constitution_path}')
      data = io_util.read_json_from_file(constitution_path)
      best_constitution_on_test_set = data['best_constitution_on_test_set']
      latest_constitution = data['latest_constitution']
      current_epoch = data['epoch'] + 1
      if os.path.exists(
          f'{self.constitution_save_path}/train_data_with_results.json'
      ):
        train_data_with_results = io_util.read_json_from_file(
            f'{self.constitution_save_path}/train_data_with_results.json'
        )
        val_data_with_results = io_util.read_json_from_file(
            f'{self.constitution_save_path}/val_data_with_results.json'
        )
        test_data_with_results = io_util.read_json_from_file(
            f'{self.constitution_save_path}/test_data_with_results.json'
        )
        all_train_losses = io_util.read_json_from_file(
            f'{self.constitution_save_path}/train_losses.json'
        )
        all_val_losses = io_util.read_json_from_file(
            f'{self.constitution_save_path}/val_losses.json'
        )
        all_test_losses = io_util.read_json_from_file(
            f'{self.constitution_save_path}/test_losses.json'
        )
    return (
        latest_constitution,
        best_constitution_on_test_set,
        current_epoch,
        train_data_with_results,
        val_data_with_results,
        test_data_with_results,
        all_train_losses,
        all_val_losses,
        all_test_losses,
    )

  def extract_initial_constitution(self, dataset: list[dict[str, Any]]) -> str:
    """Extract the constitution from the LLM."""
    examples = [
        f'{e["example_desc_for_llm"]}\n'
        f'Objective Satisfied Score: {e["satisfied_score"]}'
        for e in dataset
    ]

    preamble = prompts.INITIAL_CONSTITUTION_PREAMBLE.format(
        objective=self.config.objective,
        formatted_examples='\n\n'.join(examples),
    )
    llm_prompt = f"""
{preamble}
Study the provided examples to propose a constitution of strategies.

{prompts.CONSTITUTION_PLAN.format(num_strategies=self.config.initial_num_strategies)}
"""
    constitution = self.engine_llm.generate(llm_prompt)
    return constitution

  def compute_loss(
      self, data_with_results: list[dict[str, Any]], current_epoch: int
  ) -> float:
    """Compute the loss of the surrogate classifier for the given epoch."""
    if not data_with_results:
      return MAX_LOSS

    squared_errors = []
    for example in data_with_results:
      prediction = example.get(f'satisfied_pred_{current_epoch}')
      satisfied_score = example.get('satisfied_score')
      if prediction is not None and satisfied_score is not None:
        squared_errors.append((prediction - satisfied_score) ** 2)

    mean_squared_error = sum(squared_errors) / len(squared_errors)
    return mean_squared_error

  def update_constitution_with_surrogate_feedback(
      self,
      constitution: str,
      data_with_results: list[dict[str, Any]],
      current_epoch: int,
      current_num_strategies: int,
      current_change_percentage: float,
  ) -> str:
    """Update the constitution using LLM."""
    preamble = prompts.CONSTITUTION_UPDATE_PREAMBLE.format(
        objective=self.config.objective,
        constitution=constitution,
    )
    satisfied_key = f'satisfied_pred_{current_epoch}'
    failed_examples = [
        e for e in data_with_results if e[satisfied_key] != e['satisfied_score']
    ]
    examples_str = '\n\n'.join([
        e['example_desc_for_llm']
        + f'\nGround Truth: {e["satisfied_score"]}\n'
        f'Prediction: {e[satisfied_key]}'
        for e in failed_examples
    ])
    example_prompt = f"""
We provide the ground truth and prediction for the following examples.
{examples_str}
  """
    update_plan = prompts.CONSTITUTION_UPDATE_PLAN.format(
        num_strategies=current_num_strategies,
        change_percentage=current_change_percentage,
    )

    output_plan = f"""
Follow the same rules in your updated constitution as the given constitution.
{prompts.CONSTITUTION_PLAN.format(num_strategies=current_num_strategies)}

Follow these rules to update the constitution:
{update_plan}

Respond only with the updated constitution, nothing else. No other text or chain of thought.
  """
    llm_prompt = f"""{preamble}
  {example_prompt}
  {output_plan}
  """
    try:
      response = self.engine_llm.generate(llm_prompt)
    except Exception as e:  # pylint: disable=broad-except
      print(f'Error generating updated constitution: {e}')
      response = constitution

    updated_constitution = response
    return updated_constitution

  def evaluate_example_on_surrogate_classifier(
      self, example: dict[str, Any], constitution: str, current_epoch: int
  ) -> dict[str, Any]:
    """Evaluate the examples with the constitution."""

    task = prompts.SURROGATE_CLASSIFIER_TASK.format(
        objective=self.config.objective,
        constitution=constitution,
        example=example['example_desc_for_llm'],
    )
    satisfied_pred = None
    try:
      response = self.engine_llm.generate_object(task, FloatResponse)
      satisfied_pred = response.value
    except Exception as e:  # pylint: disable=broad-except
      print(f'Failed to predict or parse satisfied_pred: {satisfied_pred}')
      print(f'example: {example}')
      print(f'Error: {e}')
      satisfied_pred = None
    example[f'satisfied_pred_{current_epoch}'] = satisfied_pred
    return example

  def evaluate_candidate_constitutions_as_surrogate_classifiers(
      self,
      data_with_results: list[dict[str, Any]],
      candidate_constitutions: list[str],
      current_epoch: int,
  ) -> tuple[list[list[dict[str, Any]]], list[float]]:
    """Evaluate the candidate constitutes as a surrogate classifier."""
    list_of_kwargs_to_function = []
    for candidate_constitution in candidate_constitutions:
      for example in data_with_results:
        list_of_kwargs_to_function.append({
            'example': example,
            'constitution': candidate_constitution,
            'current_epoch': current_epoch,
        })
    results = parallel_util.run_in_parallel(
        function=self.evaluate_example_on_surrogate_classifier,
        list_of_kwargs=list_of_kwargs_to_function,
        num_workers=min(
            interface.MAX_PARALLELISM, len(list_of_kwargs_to_function)
        ),
    )
    candidate_data_with_results = []
    candidate_losses = []
    for i in range(len(candidate_constitutions)):
      data_with_results = results[
          i * len(data_with_results) : (i + 1) * len(data_with_results)
      ]
      candidate_data_with_results.append(data_with_results)
      candidate_losses.append(
          self.compute_loss(data_with_results, current_epoch)
      )
    return candidate_data_with_results, candidate_losses

  def run_optimizer(self) -> str:
    """Run the constitution optimizer."""
    (
        latest_constitution,
        best_constitution_on_test_set,
        current_epoch,
        train_data_with_results,
        val_data_with_results,
        test_data_with_results,
        all_train_losses,
        all_val_losses,
        all_test_losses,
    ) = self.load_checkpoint()

    best_val_loss = MAX_LOSS
    if all_val_losses:
      best_val_loss = min(all_val_losses)

    best_test_loss = MAX_LOSS
    if all_test_losses:
      best_test_loss = min(all_test_losses)

    if not latest_constitution:
      if self.config.initial_constitution:
        latest_constitution = self.config.initial_constitution
      else:
        latest_constitution = self.extract_initial_constitution(
            random.sample(
                train_data_with_results,
                min(self.config.batch_size, len(train_data_with_results)),
            )
        )

    candidate_constitutions = [latest_constitution]

    for epoch in range(current_epoch, self.config.epochs):
      epoch_num_strategies = eval_util.get_current_decay_value(
          initial_value=self.config.initial_num_strategies,
          final_value=self.config.final_num_strategies,
          current_epoch=epoch,
          epochs=self.config.epochs,
      )

      epoch_allowed_change_percentage = eval_util.get_current_decay_value(
          initial_value=self.config.initial_change_percentage,
          final_value=self.config.final_change_percentage,
          current_epoch=epoch,
          epochs=self.config.epochs,
      )
      print(
          f'Epoch {epoch}: num_strategies={epoch_num_strategies},'
          f' allowed_change_percentage={epoch_allowed_change_percentage}',
      )

      random.shuffle(train_data_with_results)
      if epoch != 0:
        print(
            'Get candidate constitutions by updating the latest constitution'
            ' with surrogate feedback.'
        )
        candidate_constitutions = parallel_util.run_in_parallel(
            function=self.update_constitution_with_surrogate_feedback,
            list_of_kwargs=[
                {
                    'constitution': latest_constitution,
                    'data_with_results': train_data_with_results[
                        i : i + self.config.batch_size
                    ],
                    'current_epoch': epoch - 1,
                    'current_num_strategies': epoch_num_strategies,
                    'current_change_percentage': (
                        epoch_allowed_change_percentage
                    ),
                }
                for i in range(
                    0, len(train_data_with_results), self.config.batch_size
                )
            ],
            num_workers=min(
                interface.MAX_PARALLELISM,
                (len(train_data_with_results) // self.config.batch_size) + 1,
            ),
        )

      print(
          f'Evaluating val data against {len(candidate_constitutions)}'
          ' candidate constitutions.'
      )
      candidate_val_data_with_results, candidate_val_losses = (
          self.evaluate_candidate_constitutions_as_surrogate_classifiers(
              data_with_results=val_data_with_results,
              candidate_constitutions=candidate_constitutions,
              current_epoch=epoch,
          )
      )
      latest_constitution_index = np.argmin(candidate_val_losses)
      latest_constitution = candidate_constitutions[latest_constitution_index]
      val_data_with_results = candidate_val_data_with_results[
          latest_constitution_index
      ]
      val_loss = candidate_val_losses[latest_constitution_index]
      all_val_losses.append(val_loss)

      if val_loss <= best_val_loss:
        best_val_loss = val_loss
        print(
            f'New best constitution found at epoch {epoch} with val loss:'
            f' {best_val_loss}'
        )

      print('Evaluating train data against latest constitution.')
      train_data_with_results, train_loss = (
          self.evaluate_candidate_constitutions_as_surrogate_classifiers(
              data_with_results=train_data_with_results,
              candidate_constitutions=[latest_constitution],
              current_epoch=epoch,
          )
      )
      train_data_with_results, train_loss = (
          train_data_with_results[0],
          train_loss[0],
      )
      all_train_losses.append(train_loss)

      print('Evaluating test data against latest constitution.')
      test_data_with_results, test_loss = (
          self.evaluate_candidate_constitutions_as_surrogate_classifiers(
              data_with_results=test_data_with_results,
              candidate_constitutions=[latest_constitution],
              current_epoch=epoch,
          )
      )
      test_data_with_results, test_loss = (
          test_data_with_results[0],
          test_loss[0],
      )
      all_test_losses.append(test_loss)

      if test_loss <= best_test_loss:
        best_test_loss = test_loss
        print(
            f'New best constitution found at epoch {epoch} with test loss:'
            f' {best_test_loss}'
        )
        print(f'Best constitution:\n{latest_constitution}')
        best_constitution_on_test_set = latest_constitution

      print(f'Epoch: {epoch}')
      print(f'Train loss: {train_loss}')
      print(f'Val loss: {val_loss}')
      print(f'Test loss: {test_loss}')
      print(f'Best val loss: {best_val_loss}')
      print(f'Best test loss: {best_test_loss}')

      self.save_checkpoint(
          latest_constitution=latest_constitution,
          best_constitution_on_test_set=best_constitution_on_test_set,
          current_epoch=epoch,
          train_data_with_results=train_data_with_results,
          val_data_with_results=val_data_with_results,
          test_data_with_results=test_data_with_results,
          all_train_losses=all_train_losses,
          all_val_losses=all_val_losses,
          all_test_losses=all_test_losses,
      )
    return best_constitution_on_test_set
