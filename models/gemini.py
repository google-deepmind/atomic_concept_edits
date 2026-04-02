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

"""Gemini model for ACE."""

from concurrent import futures
import dataclasses
import os
from typing import TypeVar

from google import genai
from google.genai import types
import pydantic
import tenacity

from .. import interface

T = TypeVar('T')

RETRY_WAIT_SECONDS = 1
RETRY_MAX_ATTEMPTS = 1


@dataclasses.dataclass
class GeminiModelConfig(interface.ModelConfig):
  """Configuration for a Gemini model using the public google-genai SDK."""

  model: str
  api_key: str

  def build_model(self) -> interface.Model:
    return GeminiModel(self)


class GeminiModel(interface.Model):
  """Model using the public google-genai SDK."""

  def __init__(self, config: GeminiModelConfig):
    self.config = config
    self.client = genai.Client(api_key=config.api_key)

  @tenacity.retry(
      stop=tenacity.stop_after_attempt(RETRY_MAX_ATTEMPTS),
      wait=tenacity.wait_fixed(RETRY_WAIT_SECONDS)
      + tenacity.wait_random(0, RETRY_WAIT_SECONDS),
  )
  def generate(
      self,
      prompt: interface.Content,
      load_response: bool = True,
      num_responses: int = 1,
      save_path: str | None = None,
  ) -> interface.Content:
    """Generates text responses from the model."""
    if not isinstance(prompt, str):
      if isinstance(prompt, list):
        prompt = '\n'.join(str(p) for p in prompt)
      else:
        prompt = str(prompt)

    with futures.ThreadPoolExecutor(max_workers=num_responses) as executor:
      results = []
      for _ in range(num_responses):
        results.append(
            executor.submit(
                self.client.models.generate_content,
                model=self.config.model,
                contents=prompt,
            )
        )
      executor.shutdown(wait=True)

    responses = []
    for result in results:
      response = result.result()
      responses.append(response.text)

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

  @tenacity.retry(
      stop=tenacity.stop_after_attempt(RETRY_MAX_ATTEMPTS),
      wait=tenacity.wait_fixed(RETRY_WAIT_SECONDS)
      + tenacity.wait_random(0, RETRY_WAIT_SECONDS),
  )
  def generate_object(
      self, prompt: interface.Content, cls: pydantic.BaseModel
  ) -> pydantic.BaseModel:
    """Generates a structured object from the model."""
    if not isinstance(prompt, str):
      prompt = str(prompt)

    result = self.client.models.generate_content(
        model=self.config.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_json_schema=cls.model_json_schema(),
        ),
    )
    if result.text is None:
      raise ValueError('Model returned empty or non-text response.')
    response = result.text.replace('```json', '').replace('```', '').strip()
    return cls.model_validate_json(response)
