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

"""Tree utility functions for ACE generation."""

import os
import random
from typing import Sequence
import uuid

import pandas as pd

from .. import ace_set_class
from .. import data
from .. import interface
from .. import mutation_samplers
from . import parallel_util


class PromptNode:
  """A node representing a prompt in the exploration tree."""

  def __init__(
      self,
      prompt: str,
      parent: 'PromptNode | None' = None,
      ace: ace_set_class.ACE | None = None,
      depth: int = 0,
      prompt_id: str | None = None,
  ):
    """Initializes the prompt node.

    Args:
      prompt: The prompt text.
      parent: The parent node of the prompt.
      ace: The ACE that was used to generate the prompt from the parent.
      depth: The depth number of the prompt in the exploration tree.
      prompt_id: The ID of the prompt.
    """
    self.prompt = prompt
    self.parent = parent
    self.ace = ace
    self.depth = depth
    self.children = []
    if prompt_id is None:
      self.prompt_id = str(uuid.uuid4())
    else:
      self.prompt_id = prompt_id
    if self.parent is None:
      self.root = self
    else:
      self.root = self.parent.root

  def save_to_df(
      self,
      df: pd.DataFrame,
      target_model_responses_column: str = 'target_model_responses',
      target_model_response_paths_column: str = 'target_model_response_paths',
      autorater_scores_column: str = 'autorater_scores',
      objective_satisfied_column: str = 'objective_satisfied',
  ) -> pd.DataFrame:
    """Saves the node to a dataframe."""
    row = {
        'prompt_id': self.prompt_id,
        'depth': self.depth,
        'prompt': self.prompt,
        'parent_id': self.parent.prompt_id if self.parent else None,
        'root_id': self.root.prompt_id if self.root else None,
        'parent_prompt': self.parent.prompt if self.parent else None,
        'root_prompt': self.root.prompt if self.root else None,
        'ace_verbalization': self.ace.verbalization if self.ace else None,
        'ace_score': self.ace.ace_score if self.ace else None,
        target_model_responses_column: pd.NA,
        target_model_response_paths_column: pd.NA,
        autorater_scores_column: pd.NA,
        objective_satisfied_column: pd.NA,
    }
    if self.prompt_id not in df['prompt_id']:
      df.loc[len(df)] = row
    else:
      print(
          f'Warning: Prompt {self.prompt_id} already exists in the dataframe.'
          f' Skipping duplicate prompt {self.prompt}.'
      )
    return df

  def add_child(self, child: 'PromptNode'):
    self.children.append(child)


def generate_mutations_for_nodes(
    nodes: list[PromptNode],
    mutation_sampler: interface.MutationSampler,
    sample_size_at_depth: Sequence[int],
    max_parallelism: int = interface.MAX_PARALLELISM,
) -> list[interface.Content | None]:
  """Generates actions in parallel for a list of nodes.

  Args:
    nodes: A list of nodes to generate actions for.
    mutation_sampler: The mutation sampler to use for generating actions.
    sample_size_at_depth: The number of actions to sample at each depth.
    max_parallelism: The maximum number of parallel sampling calls to make.

  Returns:
    A list of concept bags, one for each node.
  """

  all_generated_mutations = []
  if nodes:
    print(f'  Generating actions for {len(nodes)} prompts')
    sampling_function = mutation_sampler.sample
    if isinstance(mutation_sampler, mutation_samplers.ACEMutationSampler):
      sampling_function = mutation_sampler.sample_aces

    list_of_kwargs = [
        {
            'prompt': node.prompt,
            'num_samples': sample_size_at_depth[node.depth],
        }
        for node in nodes
    ]
    all_generated_mutations = parallel_util.run_in_parallel(
        function=sampling_function,
        list_of_kwargs=list_of_kwargs,
        num_workers=min(max_parallelism, len(nodes)),
    )
  return all_generated_mutations


def build_exploration_tree(
    mutation_sampler: interface.MutationSampler,
    exploration_depth: int,
    sample_size_at_depth: Sequence[int],
    save_path: str,
    dataset_config: data.DatasetConfig | None = None,
    run_id: str = '',
) -> str:
  """Build the exploration tree of prompts and queue them for evaluation.

  Args:
    mutation_sampler: The mutation sampler to use for generating actions.
    exploration_depth: The maximum depth of the exploration tree for each
      prompt.
    sample_size_at_depth: The number of actions to sample at each depth.
    save_path: The path to save the exploration data to.
    dataset_config: The dataset config to use for generating starter prompts.
    run_id: The run ID of the exploration tree (default is a random hex string).

  Returns:
    The run ID of the exploration tree.
  """

  if len(sample_size_at_depth) != exploration_depth:
    raise ValueError(
        'sample_size_at_depth must have the same length as exploration_depth.'
    )

  if not run_id:
    run_id = random.randbytes(3).hex()
  print(f'Run ID: {run_id}')
  save_exploration_data_path = os.path.join(
      save_path, run_id, 'exploration_data.csv'
  )
  print(f'Saving exploration data to: {save_exploration_data_path}')

  os.makedirs(os.path.join(save_path, run_id), exist_ok=True)

  if os.path.exists(save_exploration_data_path):
    print(f'Exploration data {save_exploration_data_path} already exists.')
    exploration_data = pd.read_csv(save_exploration_data_path)
    root_nodes = []
    for _, row in exploration_data.iterrows():
      if exploration_data[
          exploration_data['parent_id'] == row['prompt_id']
      ].empty:
        root_nodes.append(
            PromptNode(
                prompt=row['prompt'],
                depth=row['depth'],
                prompt_id=row['prompt_id'],
            )
        )

  else:
    print('\n--- Building Exploration Tree ---')
    starter_prompt_list = data.DatasetFactory()(dataset_config)
    print(f'Loaded {len(starter_prompt_list)} starter prompts')

    root_nodes = [PromptNode(prompt=p, depth=0) for p in starter_prompt_list]

    columns = {
        'prompt_id': 'string',
        'depth': 'Int64',
        'prompt': 'string',
        'parent_id': 'string',
        'parent_prompt': 'string',
        'ace_verbalization': 'string',
        'ace_score': 'Float64',
        'root_prompt': 'string',
        'root_id': 'string',
    }
    exploration_data = pd.DataFrame(columns=columns.keys()).astype(columns)
    for node in root_nodes:
      exploration_data = node.save_to_df(exploration_data)

  current_nodes = root_nodes
  print(f'Initial prompts: {[node.prompt for node in root_nodes]}')

  while current_nodes:
    print(f'  Number of nodes to expand: {len(current_nodes)}')

    all_generated_mutations = generate_mutations_for_nodes(
        nodes=current_nodes,
        mutation_sampler=mutation_sampler,
        sample_size_at_depth=sample_size_at_depth,
    )

    next_nodes = []
    for i, generated_mutations in enumerate(all_generated_mutations):
      if generated_mutations is None:
        continue
      parent_node = current_nodes[i]

      for mutation in generated_mutations:
        if isinstance(mutation, ace_set_class.ACE):
          mutated_prompt = mutation.updated_prompt
        else:
          mutated_prompt = mutation

        if any(
            child.prompt == mutated_prompt for child in parent_node.children
        ):
          print(f'      Skipping duplicate child prompt: {mutated_prompt}')
          continue
        child_node = PromptNode(
            prompt=mutated_prompt,
            parent=parent_node,
            ace=mutation if isinstance(mutation, ace_set_class.ACE) else None,
            depth=parent_node.depth + 1,
        )
        parent_node.add_child(child_node)
        exploration_data = child_node.save_to_df(exploration_data)
        if child_node.depth < exploration_depth:
          next_nodes.append(child_node)
    exploration_data.to_csv(save_exploration_data_path, index=False)

    current_nodes = next_nodes

  if exploration_data is not None:
    exploration_data.to_csv(save_exploration_data_path, index=False)

  print(
      '--- Exploration Tree Built ---', f'saved at {save_exploration_data_path}'
  )
  return run_id


class PromptReferenceNode:
  """A node representing a prompt in the lineage tree for analysis."""

  def __init__(self, prompt_id: str, df_iloc: int | None = None):
    self.prompt_id = prompt_id
    self.root_id = None
    self.df_iloc = df_iloc
    self.parent = None
    self.children = []

  def __repr__(self):
    return f'Node({self.prompt_id[:8]}...)'


def build_lineage_trees(
    df: pd.DataFrame, use_prompt_text_as_id: bool = False
) -> list[PromptReferenceNode]:
  """Builds lineage trees from a DataFrame.

  Args:
    df: A pandas DataFrame containing prompt exploration data, including
      'prompt_id', 'parent_id', and 'root_id' columns.
    use_prompt_text_as_id: If True, uses 'prompt', 'parent_prompt', and
      'root_prompt' as IDs instead of the *_id columns.

  Returns:
    A list of PromptReferenceNode objects, each representing the root of a
    distinct lineage tree.
  """
  all_nodes = {}

  id_col = 'prompt' if use_prompt_text_as_id else 'prompt_id'
  parent_id_col = 'parent_prompt' if use_prompt_text_as_id else 'parent_id'
  root_id_col = 'root_prompt' if use_prompt_text_as_id else 'root_id'

  def get_or_create_node(pid, iloc=None):
    if pd.isna(pid) or str(pid).strip().lower() in ('', 'none'):
      return None
    if pid not in all_nodes:
      all_nodes[pid] = PromptReferenceNode(pid)
    if iloc is not None and all_nodes[pid].df_iloc is None:
      all_nodes[pid].df_iloc = iloc
    return all_nodes[pid]

  for i, (_, row) in enumerate(df.iterrows()):
    node = get_or_create_node(row[id_col], iloc=i)
    if node is None:
      continue

    node.root_id = row.get(root_id_col)

    parent_id = row.get(parent_id_col)
    if not pd.isna(parent_id) and str(parent_id).strip().lower() not in (
        '',
        'none',
    ):
      parent_node = get_or_create_node(parent_id)
      node.parent = parent_node
      parent_node = get_or_create_node(parent_id)
      node.parent = parent_node
      if node not in parent_node.children:
        parent_node.children.append(node)

  root_nodes = []
  seen_roots = set()

  for pid, node in all_nodes.items():
    current = node
    while current.parent is not None:
      current = current.parent

    if node.root_id and current.prompt_id != node.root_id:
      print(
          f'Mismatch Detected! Node {pid} expects root {node.root_id}'
          f'but found {current.prompt_id}'
      )

    if current.prompt_id not in seen_roots:
      root_nodes.append(current)
      seen_roots.add(current.prompt_id)

  return root_nodes
