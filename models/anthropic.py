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

"""Anthropic model for ACE."""

from concurrent import futures
import dataclasses
import os

import anthropic
import pydantic

from .. import interface


@dataclasses.dataclass
class AnthropicModelConfig(interface.ModelConfig):
  """Configuration for an Anthropic model."""

  model_name: str
  api_key: str
  max_length: int = 8192

  def build_model(self) -> interface.Model:
    return AnthropicModel(self)


class AnthropicModel(interface.Model):
  """Model using the public Anthropic SDK."""

  def __init__(self, config: AnthropicModelConfig):
    self.config = config
    self.client = anthropic.Anthropic(api_key=config.api_key)

  def generate(
      self,
      prompt: interface.Content,
      load_response: bool = True,
      num_responses: int = 1,
      save_path: str | None = None,
  ) -> interface.Content:
    """Generates text responses from the Anthropic model."""
    if not isinstance(prompt, str):
      prompt = str(prompt)

    with futures.ThreadPoolExecutor(max_workers=num_responses) as executor:
      results = []
      for _ in range(num_responses):
        results.append(
            executor.submit(
                self.client.messages.create,
                model=self.config.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=self.config.max_length,
            )
        )
      executor.shutdown(wait=True)

    responses = [r.result().content[0].text for r in results]

    if save_path is not None:
      os.makedirs(save_path, exist_ok=True)
      for i, response in enumerate(responses):
        text_path = os.path.join(save_path, f'response_text_{i}.txt')
        with open(text_path, 'w+') as f:
          f.write(response)

    if num_responses == 1:
      return responses[0]
    else:
      return responses

  def generate_object(
      self, prompt: interface.Content, cls: pydantic.BaseModel
  ) -> pydantic.BaseModel:
    """Generates a structured object from the model."""
    raise NotImplementedError
