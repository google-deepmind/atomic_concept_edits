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

"""ACE Exploration class that interleaves exploration and evaluation."""

import dataclasses
import glob
import json
import os
import random

import pandas as pd

from . import ace_set_class
from . import data
from . import interface
from .util import eval as eval_util
from .util import parallel_util
from .util import tree as tree_util


@dataclasses.dataclass
class ACEExplorationConfig:
  """Configuration for ACE exploration and evaluation.

  mutation_sampler_config: Configuration for the mutation sampler.
  target_model_config: Configuration for the target model. If None,
    we continue exploration without generating target model responses.
  autorater_config: Configuration for the autorater. If None, we continue
    exploration without generating autorater scores.
  scorer_config: Configuration for a scorer. When provided,
    target_model_config and autorater_config are ignored and
    generate_and_score is used instead.
  dataset_config: Configuration for the initial prompt dataset.
  initial_prompts: List of initial prompts. Use either this or dataset_config.
  sample_size_at_depth: The number of mutated samples to generate for
    each node at each depth level of the exploration tree.
  num_responses_per_prompt: The number of responses to generate for each
    prompt.
  min_score: The minimum score for the autorater that is considered as
    objective successful.
  max_score: The maximum score for the autorater that is considered as
    objective successful.
  score_selection_method: The autorater score selection method to use when
    num_responses_per_prompt > 1.
  max_parallelism_target_model: The max number of parallel sampling calls to
    make for target model.
  max_parallelism_autorater: The max number of parallel sampling calls to make
    for autorater.
  max_parallelism_ace_generation: The max number of parallel sampling calls to
    make for ACE generation.
  save_path: The path to save the results to.
  run_id: The run ID of the exploration.
  save_target_model_responses: Whether to save the target model responses.
  target_model_responses_column: The name of the column to save the target
    model responses in.
  target_model_response_paths_column: The name of the column to save the target
    model response paths in.
  autorater_scores_column: The name of the column to save the autorater scores
    in.
  objective_satisfied_column: The name of the column to save the objective
    satisfied boolean in.
  """

  mutation_sampler_config: interface.MutationSamplerConfig
  target_model_config: interface.ModelConfig | None = None
  autorater_config: interface.AutoraterConfig | None = None
  scorer_config: interface.ScorerConfig | None = None
  dataset_config: data.DatasetConfig | None = None
  initial_prompts: list[str] | None = None
  sample_size_at_depth: tuple[int, ...] = (5, 3, 2)
  num_responses_per_prompt: int = 1
  min_score: float = 0.0
  max_score: float = 1.0
  score_selection_method: eval_util.AutoraterScoreSelectionMethod = (
      eval_util.AutoraterScoreSelectionMethod.ANY
  )
  max_parallelism_target_model: int = interface.MAX_PARALLELISM
  max_parallelism_autorater: int = interface.MAX_PARALLELISM
  max_parallelism_ace_generation: int = interface.MAX_PARALLELISM
  save_path: str = '/tmp/ace/'
  run_id: str = ''
  save_target_model_responses: bool = False
  target_model_responses_column: str = interface.TARGET_MODEL_RESPONSES_COLUMN
  target_model_response_paths_column: str = (
      interface.TARGET_MODEL_RESPONSE_PATHS_COLUMN
  )
  autorater_scores_column: str = interface.AUTORATER_SCORES_COLUMN
  objective_satisfied_column: str = interface.OBJECTIVE_SATISFIED_COLUMN


class ACEExploration:
  """Interleaves ACE exploration and evaluation using a data-driven loop.

  The flow is:
    1. Load initial prompts (or depth=0 rows from existing dataframe).
    2. Run target model if model responses are not available.
    3. Run autorater if autorater scores are not available.
    4. Select nodes to sample children (ACE) for.
    5. Sample children for these nodes if not already sampled. If some
       children already exist, sample only remaining children up to the
       given sample_size at that depth.
    6. Queue all pending nodes (new and existing children) and repeat
       from step 2.
  """

  def __init__(
      self,
      config: ACEExplorationConfig,
  ):
    self._config = config

    self._mutation_sampler = (
        config.mutation_sampler_config.build_mutation_sampler()
    )
    if config.scorer_config is not None:
      self.scorer = config.scorer_config.build_scorer()
      self.target_model = None
      self.autorater = None
    else:
      self.scorer = None
      self.target_model = (
          config.target_model_config.build_model()
          if config.target_model_config is not None
          else None
      )
      self.autorater = (
          config.autorater_config.build_autorater()
          if config.autorater_config is not None
          else None
      )
    self._init_run_id()
    self._exploration_depth = len(self._config.sample_size_at_depth)

    columns = {
        interface.PROMPT_ID_COLUMN: 'string',
        interface.DEPTH_COLUMN: 'Int64',
        interface.PROMPT_COLUMN: 'string',
        interface.PARENT_ID_COLUMN: 'string',
        interface.PARENT_PROMPT_COLUMN: 'string',
        interface.ACE_VERBALIZATION_COLUMN: 'string',
        interface.ACE_SCORE_COLUMN: 'Float64',
        interface.ROOT_PROMPT_COLUMN: 'string',
        interface.ROOT_ID_COLUMN: 'string',
        config.target_model_responses_column: 'string',
        config.target_model_response_paths_column: 'string',
        config.autorater_scores_column: 'string',
        config.objective_satisfied_column: 'boolean',
    }
    self.exploration_data = pd.DataFrame(columns=columns.keys()).astype(columns)

    if os.path.exists(self.exploration_data_path):
      print(f'Resuming from existing data: {self.exploration_data_path}')
      self.exploration_data = pd.read_csv(self.exploration_data_path)
    else:
      if self._config.initial_prompts is not None:
        prompts = self._config.initial_prompts
      elif self._config.dataset_config is not None:
        prompts = data.DatasetFactory()(self._config.dataset_config)
      else:
        raise ValueError(
            'Either initial_prompts or dataset_config must be provided.'
        )
      print(f'Loaded {len(prompts)} initial prompts')

      root_nodes = [tree_util.PromptNode(prompt=p, depth=0) for p in prompts]
      for node in root_nodes:
        self.exploration_data = node.save_to_df(
            self.exploration_data,
            target_model_responses_column=self._config.target_model_responses_column,
            target_model_response_paths_column=self._config.target_model_response_paths_column,
            autorater_scores_column=self._config.autorater_scores_column,
            objective_satisfied_column=self._config.objective_satisfied_column,
        )

    self._save_exploration_data()

  @property
  def exploration_data_path(self) -> str:
    return os.path.join(
        self._config.save_path, self._config.run_id, 'exploration_data.csv'
    )

  def _init_run_id(self):
    if not self._config.run_id:
      self._config.run_id = random.randbytes(3).hex()
    print(f'Run ID: {self._config.run_id}')
    run_dir = os.path.join(self._config.save_path, self._config.run_id)
    os.makedirs(run_dir, exist_ok=True)

  def _save_exploration_data(self):
    self.exploration_data.to_csv(self.exploration_data_path, index=False)

  def _get_nodes_at_depth(self, depth: int) -> list[tree_util.PromptNode]:
    """Get rows at given depth and prepare nodes."""
    rows = self.exploration_data[
        self.exploration_data[interface.DEPTH_COLUMN] == depth
    ]
    nodes = []
    for _, row in rows.iterrows():
      parent_row = self.exploration_data[
          self.exploration_data[interface.PROMPT_ID_COLUMN]
          == row[interface.PARENT_ID_COLUMN]
      ]
      if parent_row.empty:
        parent_node = None
      else:
        parent_node = tree_util.PromptNode(
            prompt=parent_row[interface.PROMPT_COLUMN].values[0],
            depth=int(parent_row[interface.DEPTH_COLUMN].values[0]),
            prompt_id=parent_row[interface.PROMPT_ID_COLUMN].values[0],
        )
        parent_node.root = tree_util.PromptNode(
            prompt=parent_row[interface.ROOT_PROMPT_COLUMN].values[0],
            depth=0,
            prompt_id=parent_row[interface.ROOT_ID_COLUMN].values[0],
        )
      nodes.append(
          tree_util.PromptNode(
              prompt=row[interface.PROMPT_COLUMN],
              depth=int(row[interface.DEPTH_COLUMN]),
              prompt_id=row[interface.PROMPT_ID_COLUMN],
              parent=parent_node,
          )
      )
    return nodes

  def _run_target_model(self, nodes: list[tree_util.PromptNode]):
    """Generates responses for nodes that are missing target model responses."""

    nodes_to_generate = []
    for node in nodes:
      row = self.exploration_data[
          self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      ]
      if pd.isna(
          row[self._config.target_model_responses_column].values[0]
      ) and pd.isna(
          row[self._config.target_model_response_paths_column].values[0]
      ):
        nodes_to_generate.append(node)

    if not nodes_to_generate:
      return

    print(f'\n--- Running target model on {len(nodes_to_generate)} prompts ---')

    list_of_kwargs = []
    for node in nodes_to_generate:
      save_path = None
      if self._config.save_target_model_responses:
        save_path = os.path.join(
            self._config.save_path,
            self._config.run_id,
            'target_model_responses',
            node.prompt_id,
        )
      list_of_kwargs.append({
          'target_model': self.target_model,
          'prompt': node.prompt,
          'num_responses': self._config.num_responses_per_prompt,
          'save_path_for_responses': save_path,
      })
    results = parallel_util.run_in_parallel(
        function=eval_util.generate_target_model_response,
        list_of_kwargs=list_of_kwargs,
        num_workers=min(
            self._config.max_parallelism_target_model, len(list_of_kwargs)
        ),
    )

    for j, responses in enumerate(results):
      node = nodes_to_generate[j]
      mask = self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      if self._config.save_target_model_responses:
        response_paths = glob.glob(
            os.path.join(
                self._config.save_path,
                self._config.run_id,
                'target_model_responses',
                node.prompt_id,
            )
            + '/*'
        )
        if response_paths:
          self.exploration_data.loc[
              mask, self._config.target_model_response_paths_column
          ] = json.dumps(response_paths)
      else:
        if responses is not None:
          self.exploration_data.loc[
              mask, self._config.target_model_responses_column
          ] = json.dumps(responses)

    self._save_exploration_data()

  def _run_scorer(self, nodes: list[tree_util.PromptNode]):
    """Generates responses and scores using combined model+autorater."""

    list_of_kwargs = []
    nodes_to_generate = []
    for node in nodes:
      row = self.exploration_data[
          self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      ]
      autorater_scores = row[self._config.autorater_scores_column].values[0]
      if pd.notna(autorater_scores):
        continue
      nodes_to_generate.append(node)
      save_path = None
      if self._config.save_target_model_responses:
        save_path = os.path.join(
            self._config.save_path,
            self._config.run_id,
            'target_model_responses',
            node.prompt_id,
        )
      row = self.exploration_data[
          self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      ]
      list_of_kwargs.append({
          'prompt': node.prompt,
          'num_responses': self._config.num_responses_per_prompt,
          'save_path': save_path,
          'root_prompt': row[interface.ROOT_PROMPT_COLUMN].values[0],
      })
    if not nodes_to_generate:
      return

    print(f'\n--- Running scorer on {len(nodes_to_generate)} prompts ---')

    if self.scorer is None:
      raise ValueError('scorer is None')
    results = parallel_util.run_in_parallel(
        function=self.scorer.generate_and_score,
        list_of_kwargs=list_of_kwargs,
        num_workers=min(
            self._config.max_parallelism_target_model, len(list_of_kwargs)
        ),
    )

    for j, (responses, scores) in enumerate(results):
      node = nodes_to_generate[j]
      mask = self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id

      if self._config.save_target_model_responses:
        response_paths = glob.glob(
            os.path.join(
                self._config.save_path,
                self._config.run_id,
                'target_model_responses',
                node.prompt_id,
            )
            + '/*'
        )
        if response_paths:
          self.exploration_data.loc[
              mask, self._config.target_model_response_paths_column
          ] = json.dumps(response_paths)
      else:
        if responses is not None:
          self.exploration_data.loc[
              mask, self._config.target_model_responses_column
          ] = json.dumps(responses)

      self.exploration_data.loc[mask, self._config.autorater_scores_column] = (
          json.dumps(scores)
      )
      objective_satisfied = eval_util.is_objective_satisfied(
          autorater_scores=scores,
          min_score=self._config.min_score,
          max_score=self._config.max_score,
          selection_method=self._config.score_selection_method,
      )
      self.exploration_data.loc[
          mask, self._config.objective_satisfied_column
      ] = objective_satisfied

    self._save_exploration_data()

  def _run_autorater(self, nodes: list[tree_util.PromptNode]):
    """Scores responses for nodes that have responses but missing scores."""

    list_of_kwargs = []
    nodes_to_score = []
    for node in nodes:
      row = self.exploration_data[
          self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      ]
      responses = row[self._config.target_model_responses_column].values[0]
      response_paths = row[
          self._config.target_model_response_paths_column
      ].values[0]
      autorater_scores = row[self._config.autorater_scores_column].values[0]
      if (pd.notna(responses) or pd.notna(response_paths)) and pd.isna(
          autorater_scores
      ):
        responses = None if pd.isna(responses) else json.loads(responses)
        response_paths = (
            None if pd.isna(response_paths) else json.loads(response_paths)
        )

        nodes_to_score.append(node)
        list_of_kwargs.append({
            'autorater': self.autorater,
            'prompt': node.prompt,
            'root_prompt': row[interface.ROOT_PROMPT_COLUMN],
            'responses': responses,
            'response_paths': response_paths,
            'num_responses_per_prompt': self._config.num_responses_per_prompt,
        })

    if not list_of_kwargs:
      return

    print(f'\n--- Running autorater on {len(list_of_kwargs)} prompts ---')

    results = parallel_util.run_in_parallel(
        function=eval_util.generate_autorater_scores,
        list_of_kwargs=list_of_kwargs,
        num_workers=min(
            self._config.max_parallelism_autorater, len(list_of_kwargs)
        ),
    )

    for j, scores in enumerate(results):
      node = nodes_to_score[j]
      mask = self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      self.exploration_data.loc[mask, self._config.autorater_scores_column] = (
          json.dumps(scores)
      )
      objective_satisfied = eval_util.is_objective_satisfied(
          autorater_scores=scores,
          min_score=self._config.min_score,
          max_score=self._config.max_score,
          selection_method=self._config.score_selection_method,
      )
      self.exploration_data.loc[
          mask, self._config.objective_satisfied_column
      ] = objective_satisfied

    self._save_exploration_data()

  def _explore(
      self, nodes: list[tree_util.PromptNode], depth: int
  ) -> list[tree_util.PromptNode]:
    """Explores failed nodes by generating ACE mutations."""

    nodes_to_explore = []
    for node in nodes:
      row = self.exploration_data[
          self.exploration_data[interface.PROMPT_ID_COLUMN] == node.prompt_id
      ]
      obj_val = row[self._config.objective_satisfied_column].values[0]
      has_autorater = self.autorater is not None or self.scorer is not None
      if has_autorater and (pd.isna(obj_val) or obj_val == True):  # pylint: disable=singleton-comparison
        continue

      existing_children = self.exploration_data[
          self.exploration_data[interface.PARENT_ID_COLUMN] == node.prompt_id
      ]
      num_remaining = self._config.sample_size_at_depth[depth] - len(
          existing_children
      )
      if num_remaining > 0:
        nodes_to_explore.append(node)

    if not nodes_to_explore:
      return []

    print(
        f'\n--- Exploring {len(nodes_to_explore)} prompts at depth {depth} ---'
    )

    all_generated_mutations = tree_util.generate_mutations_for_nodes(
        nodes=nodes_to_explore,
        mutation_sampler=self._mutation_sampler,
        sample_size_at_depth=self._config.sample_size_at_depth,
        max_parallelism=self._config.max_parallelism_ace_generation,
    )

    new_children = []
    for i, generated_mutations in enumerate(all_generated_mutations):
      if generated_mutations is None:
        continue
      parent_node = nodes_to_explore[i]

      existing_child_prompts = set(
          self.exploration_data[
              self.exploration_data[interface.PARENT_ID_COLUMN]
              == parent_node.prompt_id
          ][interface.PROMPT_COLUMN].values
      )

      for mutation in generated_mutations:
        if isinstance(mutation, ace_set_class.ACE):
          mutated_prompt = mutation.updated_prompt
        else:
          mutated_prompt = mutation

        if mutated_prompt in existing_child_prompts:
          print(f'      Skipping duplicate child prompt: {mutated_prompt}')
          continue

        child_node = tree_util.PromptNode(
            prompt=mutated_prompt,
            parent=parent_node,
            ace=mutation if isinstance(mutation, ace_set_class.ACE) else None,
            depth=parent_node.depth + 1,
        )
        parent_node.add_child(child_node)
        self.exploration_data = child_node.save_to_df(
            self.exploration_data,
            target_model_responses_column=self._config.target_model_responses_column,
            target_model_response_paths_column=self._config.target_model_response_paths_column,
            autorater_scores_column=self._config.autorater_scores_column,
            objective_satisfied_column=self._config.objective_satisfied_column,
        )
        new_children.append(child_node)

    self._save_exploration_data()
    return new_children

  def run(self) -> str:
    """Runs the interleaved exploration-evaluation loop.

    Returns:
      The run ID.
    """

    for depth in range(self._exploration_depth + 1):
      print(f'\n=== Depth {depth} ===')

      nodes_at_depth = self._get_nodes_at_depth(depth)
      if not nodes_at_depth:
        print('No nodes at this depth. Stopping early.')
        break

      if self.scorer is not None:
        self._run_scorer(nodes_at_depth)
      else:
        if self.target_model is not None:
          self._run_target_model(nodes_at_depth)
        if self.autorater is not None:
          self._run_autorater(nodes_at_depth)

      if depth < self._exploration_depth:
        self._explore(nodes_at_depth, depth)

    if (
        self.exploration_data is not None
        and self.exploration_data[self._config.objective_satisfied_column]
        .notna()
        .any()
    ):
      self.exploration_data = eval_util.score_all_nodes(
          self.exploration_data,
          objective_satisfied_column=self._config.objective_satisfied_column,
      )
      self._save_exploration_data()
      eval_util.compute_metrics(
          exploration_data=self.exploration_data,
          sample_size_at_depth=self._config.sample_size_at_depth,
          save_metrics_path=os.path.join(
              self._config.save_path, self._config.run_id
          ),
          objective_satisfied_column=self._config.objective_satisfied_column,
      )

    print(
        '\n--- ACEExploration Complete --- saved at'
        f' {self.exploration_data_path}'
    )
    return self._config.run_id
