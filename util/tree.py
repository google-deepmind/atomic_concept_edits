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

from typing import Sequence
import uuid

import pandas as pd

from .. import ace_set_class
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
        interface.ACE_CONSTITUTION_STRATEGY_COLUMN: (
            str(self.ace.associated_strategy_from_constitution)
            if self.ace and self.ace.associated_strategy_from_constitution
            else None
        ),
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
