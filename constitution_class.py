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

"""Data classes for the ACE Constitution."""

from typing import Any

import pydantic


class Strategy(pydantic.BaseModel):
  """A single strategy in the constitution.

  Attributes:
    name: A short name for the strategy.
    description: A concise description (2-3 sentences) of the strategy.
  """

  name: str
  description: str

  def __str__(self) -> str:
    return f'{self.name}: {self.description}'

  def __repr__(self) -> str:
    return self.__str__()

  def to_json(self) -> dict[str, str]:
    """Converts the strategy to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, str]) -> 'Strategy':
    """Loads a strategy from a JSON dictionary."""
    return cls.model_validate(input_dict)


class Constitution(pydantic.BaseModel):
  """A structured constitution of strategies for ACE optimization.

  A constitution contains two groups of strategies:
  - Strategies most likely to be effective at satisfying the objective.
  - Strategies least likely to be effective (useful for learning boundaries).

  Attributes:
    effective_strategies: Strategies most likely to be effective.
    ineffective_strategies: Strategies least likely to be effective.
  """

  effective_strategies: list[Strategy]
  ineffective_strategies: list[Strategy]

  @property
  def num_strategies(self) -> int:
    """Returns the total number of strategies."""
    return len(self.effective_strategies) + len(self.ineffective_strategies)

  @property
  def all_strategies(self) -> list[Strategy]:
    """Returns all strategies (effective + ineffective)."""
    return self.effective_strategies + self.ineffective_strategies

  def to_string(self) -> str:
    """Renders the constitution as a human-readable string."""
    lines = ['Strategies Most Likely to be Effective:']
    for i, s in enumerate(self.effective_strategies, 1):
      lines.append(f'  Strategy {i}: {s.name} - {s.description}')
    lines.append('')
    lines.append('Strategies Least Likely to be Effective:')
    for i, s in enumerate(self.ineffective_strategies, 1):
      lines.append(f'  Strategy {i}: {s.name} - {s.description}')
    return '\n'.join(lines)

  def __str__(self) -> str:
    return self.to_string()

  def __repr__(self) -> str:
    return self.__str__()

  def to_json(self) -> dict[str, Any]:
    """Converts the constitution to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]) -> 'Constitution':
    """Loads a constitution from a JSON dictionary."""
    return cls.model_validate(input_dict)
