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

"""Autoraters for ACE."""

import dataclasses
import json
import math
import os
import re
from typing import Any

import sympy as sp

from . import interface


def _load_response_from_path(
    response_path: str,
) -> interface.Content | None:
  """Loads a text response from a file path."""
  if not os.path.exists(response_path):
    return None
  lower_path = response_path.lower()
  if lower_path.endswith(interface.TEXT_EXTENSIONS):
    with open(response_path, 'r') as f:
      return f.read()
  return None


def parse_csp(csp_string: str) -> tuple[list[tuple[str, str]], str]:
  """Parses a CSP string into equations and a target variable.

  Args:
    csp_string: The Constraint Satisfaction Problem as a string.

  Returns:
    A tuple containing:
      - A list of tuples, where each inner tuple represents an equation
        (LHS, RHS) as strings.
      - The target variable as a string.

  Raises:
    ValueError: If the target variable question is not found in the input
    string.
  """
  lines = [
      line.strip() for line in csp_string.strip().split('\n') if line.strip()
  ]
  if not lines:
    raise ValueError('CSP string is empty.')

  equations = []
  question_pattern = re.compile(r'What is ([A-Za-z_][A-Za-z0-9_]*)\?')

  for line in lines[:-1]:
    if '=' not in line:
      raise ValueError(f'Line "{line}" is not an equation.')
    parts = line.split('=', 1)
    lhs = parts[0].strip()
    rhs = parts[1].strip()
    equations.append((lhs, rhs))

  last_line = lines[-1]
  if '=' in last_line:
    raise ValueError(
        f'Last line "{last_line}" should be a question, not an equation.'
    )
  match = question_pattern.search(last_line)
  if match:
    target_variable = match.group(1)
  else:
    raise ValueError(f'Last line "{last_line}" is not a question.')

  if not target_variable:
    raise ValueError(
        'Could not find the target variable question in the input string.'
    )
  return equations, target_variable


def solve_csp(equations: list[tuple[str, str]], target_variable: str) -> float:
  """Solves a set of equations for a target variable using symbolic mathematics.

  Args:
    equations: A list of tuples, where each inner tuple represents an equation
      (LHS, RHS) as strings.
    target_variable: The variable to solve for.

  Returns:
    The numerical solution for the target variable as a float.

  Raises:
    ValueError: If the target variable cannot be solved from the given
    equations.
  """
  all_var_names = set()
  for lhs_str, rhs_str in equations:
    for s in [lhs_str, rhs_str]:
      potential_vars = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', s)
      for pv in potential_vars:
        all_var_names.add(pv)

  if not all_var_names:
    raise ValueError('No variables found in equations.')

  symbol_map = {name: sp.Symbol(name) for name in all_var_names}

  sympy_eqs = []
  for lhs_str, rhs_str in equations:
    try:
      lhs = sp.sympify(lhs_str, locals=symbol_map)
      rhs = sp.sympify(rhs_str, locals=symbol_map)
      sympy_eqs.append(sp.Eq(lhs, rhs))
    except (sp.SympifyError, TypeError) as e:
      raise ValueError(
          f'Failed to parse equation: {lhs_str}={rhs_str}: {e}'
      ) from e

  target_sym = symbol_map.get(target_variable)
  if not target_sym:
    raise ValueError(
        f"Target variable '{target_variable}' not found in equations."
    )

  try:
    solution = sp.solve(sympy_eqs, tuple(symbol_map.values()), dict=True)
  except NotImplementedError as e:
    raise ValueError(
        f'Sympy does not support solving this system of equations: {e}'
    ) from e
  except Exception as e:
    raise ValueError(f'Sympy failed to solve equations: {e}') from e

  if not solution:
    raise ValueError(f'No solution found for {target_variable}.')

  result = solution[0][target_sym]

  if result is None:
    raise ValueError(
        f"Could not solve for '{target_variable}' from solution: {solution}"
    )
  result = sp.simplify(result)
  if not result.is_number:
    raise ValueError(
        f"Could not solve for target variable '{target_variable}'. Solution is"
        f' not numeric: {result}'
    )
  return float(result.evalf())


LLM_AUTORATER_TEMPLATES = {
    'vqa': 'Does this figure show "{prompt}"? Please answer yes or no.',
}


class LLMAutoraterTaskType:
  """Task types for the LLM autorater."""

  VQA = 'vqa'
  CUSTOM = 'custom'


@dataclasses.dataclass
class WordCountAutoraterConfig(interface.AutoraterConfig):
  """Configuration for word count autorater."""

  target_word_count: int
  word_count_slope: float

  def build_autorater(self) -> interface.Autorater:
    return WordCountAutorater(self)


@dataclasses.dataclass
class LLMAutoraterConfig(interface.AutoraterConfig):
  """Configuration for an LLM autorater."""

  llm_config: interface.ModelConfig
  autorater_task_type: str
  system_instruction: str = ''
  use_logits: bool = False

  def build_autorater(self) -> interface.Autorater:
    return LLMAutorater(self)


@dataclasses.dataclass
class CSPAutoraterConfig(interface.AutoraterConfig):
  """Configuration for a CSP autorater."""

  llm_config: interface.ModelConfig

  def build_autorater(self) -> interface.Autorater:
    return CSPAutorater(self)


class WordCountAutorater(interface.Autorater):
  """Word count autorater."""

  def __init__(self, config: WordCountAutoraterConfig):
    self.config = config

  def score(
      self,
      prompt: interface.Content | None,
      response: interface.Content | None = None,
      response_path: str | None = None,
      root_prompt: interface.Content | None = None,
  ) -> float:
    """Scores an example."""
    if response is None and response_path is not None:
      response = _load_response_from_path(response_path)
    if not isinstance(response, str):
      raise ValueError(f'Response is not text: {response}')
    word_count = len(response.split())
    if word_count > self.config.target_word_count:
      score = (
          word_count - self.config.target_word_count
      ) * self.config.word_count_slope
    else:
      score = 0
    return float(score)


class LLMAutorater(interface.Autorater):
  """LLM autorater."""

  def __init__(self, config: LLMAutoraterConfig):
    self.config = config
    self.llm = config.llm_config.build_model()

  def score(
      self,
      prompt: interface.Content | None,
      response: interface.Content | None = None,
      response_path: str | None = None,
      root_prompt: interface.Content | None = None,
  ) -> float:
    """Scores an example."""
    if response is None and response_path is not None:
      response = _load_response_from_path(response_path)
    if response is None:
      raise ValueError('No response or valid response_path provided.')

    if isinstance(response, str):
      response_text = response
    else:
      response_text = str(response) if response is not None else ''

    if self.config.autorater_task_type == LLMAutoraterTaskType.CUSTOM:
      template = self.config.system_instruction
      if prompt is not None:
        formatted_si = template.format(
            prompt=str(prompt), response=response_text
        )
      else:
        formatted_si = template.format(response=response_text)
      autorater_prompt = formatted_si
    else:
      autorater_prompt_parts = [response_text]
      if prompt is not None:
        if isinstance(prompt, str):
          autorater_prompt_parts.append(
              LLM_AUTORATER_TEMPLATES[self.config.autorater_task_type].format(
                  prompt=prompt
              )
          )
        else:
          raise ValueError('We only support text prompts for LLM autoraters.')
      autorater_prompt = '\n'.join(autorater_prompt_parts)

    result = self.llm.generate(autorater_prompt)
    if not isinstance(result, str):
      result = str(result)
    return 1.0 if 'yes' in result.lower() else 0.0


class CSPAutorater(interface.Autorater):
  """CSP autorater."""

  def __init__(self, config: CSPAutoraterConfig):
    self.config = config
    self.llm = config.llm_config.build_model()

  def parse_llm_json(self, response_string: str) -> dict[str, Any] | None:
    """Parses a string response from an LLM into a Python dictionary.

    Args:
      response_string: The string response from the LLM.

    Returns:
      A dictionary parsed from the JSON string, or None if parsing fails.
    """
    try:
      clean_content = (
          response_string.replace('```json', '').replace('```', '').strip()
      )
      data = json.loads(clean_content)
      return data
    except json.JSONDecodeError as e:
      print(f'Error parsing JSON: {e}')
      return None

  def score(
      self,
      prompt: interface.Content | None,
      response: interface.Content | None = None,
      response_path: str | None = None,
      root_prompt: interface.Content | None = None,
  ) -> float:
    """Scores an example."""
    if prompt is None:
      return -1.0
    csp_string = str(prompt)
    try:
      equations, target_variable = parse_csp(csp_string)
      ground_truth_answer = solve_csp(equations, target_variable)
      solver_prompt = """
      You are a precision mathematical solver. Your task is to solve the given math problem and output the result strictly in a JSON format.

        Follow these rules:
        1.  **Format:** Your entire response must be a single valid JSON object. Do not include markdown formatting (like ```json ... ```) or conversational text.
        2.  **Schema:**
            * `reasoning_steps`: A string containing the reasoning steps to solve the problem.
            * `answer`: A number representing the concise final result.

        Example Output Structure:
        {
          "reasoning_steps":
            "Identify the known variables: a = 5, b = 10. Apply the formula for area: Area = a * b. Calculate: 5 * 10 = 50."
            ,
          "answer": "50"
        }
      """
      response = self.llm.generate(
          solver_prompt + '\n' + csp_string,
      )

      response_text = str(response)
      parsed_response = self.parse_llm_json(response_text)
      response_answer = str(parsed_response['answer'])

      if '/' in response_answer:
        num, den = map(float, response_answer.split('/'))
        response_float = num / den
      else:
        response_float = float(response_answer)
      score = float(not math.isclose(response_float, ground_truth_answer))
      return score
    except (
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        json.JSONDecodeError,
    ) as e:
      print(f"CSPAutorater failed for '{csp_string}...': {e}")
      return -1.0
