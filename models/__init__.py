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

"""Models for ACE."""

# pylint: disable=g-importing-member

# Gemini is always available (core dependency).
from .gemini import GeminiModel
from .gemini import GeminiModelConfig

# Optional providers are lazily imported to avoid requiring their SDKs.
try:
  from .openai import OpenAIModel  # pylint: disable=g-import-not-at-top
  from .openai import OpenAIModelConfig  # pylint: disable=g-import-not-at-top
except ImportError:
  pass

try:
  from .anthropic import AnthropicModel  # pylint: disable=g-import-not-at-top
  from .anthropic import AnthropicModelConfig  # pylint: disable=g-import-not-at-top
except ImportError:
  pass
