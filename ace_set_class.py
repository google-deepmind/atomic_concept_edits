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

"""Data classes for the entities and attributes."""

import enum
from typing import Any

import pydantic

from . import constitution_class


class Concept(pydantic.BaseModel):
  """An entity, attribute, relation, or idea in a prompt.

  For example, a rabbit, a park, a tree, color of rabbit,
  rabbit-park relation, etc.

  Attributes:
    name: a short name for the concept.
    description: a longer description of the concept.
  """

  name: str
  description: str

  def __str__(self):
    return_string = ''
    return_string += (
        f'\n\nConcept Name: {self.name}, Description: {self.description}\n'
    )
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def set_name(self, name: str):
    self.name = name

  def to_json(self) -> dict[str, Any]:
    """Converts the concept to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]) -> 'Concept':
    """Loads a concept from a JSON dictionary."""
    return cls.model_validate(input_dict)


class ACEType(enum.Enum):
  ADD_ACTION = 1
  REMOVE_ACTION = 2
  REPLACE_ACTION = 3


class ACE(pydantic.BaseModel):
  """An atomic action taken to modify a belief state and prompt.

  Attributes:
    ace_type: The type of ACE (e.g., ADD, REMOVE, REPLACE).
    verbalization: A natural language description of the ACE.
    updated_prompt: The prompt after applying this ACE.
    ace_score: A score associated with this ACE, indicating its quality or
      relevance.
    associated_concept: The Concept object this ACE is related to, if any.
    associated_strategy_from_constitution: A string describing the strategy from
      the constitution that led to this ACE, if applicable.
  """

  ace_type: ACEType
  verbalization: str
  updated_prompt: str
  ace_score: float
  associated_concept: Concept | None = None
  associated_strategy_from_constitution: constitution_class.Strategy | None = (
      None
  )

  def __str__(self):
    return_string = ''
    return_string += f'ACE Type: {self.ace_type}\n'
    return_string += f'ACE Verbalization: {self.verbalization}\n'
    return_string += f'Updated Prompt: {self.updated_prompt}\n'
    return_string += f'ACE Score: {self.ace_score}\n'
    strategy_str = (
        str(self.associated_strategy_from_constitution)
        if self.associated_strategy_from_constitution
        else None
    )
    return_string += f'Associated Strategy from Constitution: {strategy_str}\n'
    return return_string.strip()

  def to_json(self) -> dict[str, Any]:
    """Converts the action to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]) -> 'ACE':
    """Loads an ACE from a JSON dictionary."""
    return cls.model_validate(input_dict)


class ConceptSet(pydantic.BaseModel):
  """A collection of potentially related concepts.

  Attributes:
    concepts: A list of concepts as defined in the Concept class.
    prompt: The prompt that the concepts are related to.
  """

  concepts: list[Concept]
  prompt: str

  def __str__(self):
    return_string = f'Prompt: {self.prompt}\n'
    for concept in self.concepts:
      return_string += f'{concept.__str__()}\n\n'
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def set_prompt(self, prompt: str):
    self.prompt = prompt

  def to_json(
      self,
  ) -> dict[str, list[dict[str, Any]] | str]:
    """Converts the concept set to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]) -> 'ConceptSet':
    """Loads a concept set from a JSON dictionary."""
    return cls.model_validate(input_dict)


class ACESetForConcept(pydantic.BaseModel):
  """Contains a concept and a list of possible actions to apply to it.

  concept: the given concept of type Concept.
  aces: A list of `ACE` objects.
  """

  concept: Concept
  aces: list[ACE]

  def to_json(self) -> dict[str, Any]:
    """Converts the ActionBag to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]) -> 'ACESetForConcept':
    """Loads an ActionBag from a JSON dictionary."""
    return cls.model_validate(input_dict)


class ACESet(pydantic.BaseModel):
  """A collection of ACES to mutate a prompt.

  Attributes:
    aces: A list of ACEs that mutate a prompt to achieve a specific objective.
    prompt: The user prompt that the belief state is trying to answer.
  """

  aces: list[ACE]
  prompt: str

  def __str__(self):
    return_string = f'Prompt: {self.prompt}\n'
    for ace in self.aces:
      return_string += f'{ace.__str__()}\n\n'
    return return_string.strip()

  def __repr__(self):
    return self.__str__()

  def set_prompt(self, prompt: str):
    self.prompt = prompt

  def to_json(
      self,
  ) -> dict[str, list[dict[str, Any]] | str | float | list[float] | None]:
    """Converts the ACE set to a JSON dictionary."""
    return self.model_dump(mode='json')

  @classmethod
  def load_from_json(cls, input_dict: dict[str, Any]) -> 'ACESet':
    """Loads an ACE set from a JSON dictionary."""
    return cls.model_validate(input_dict)
