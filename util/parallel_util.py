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

"""Parallel execution utilities for ACE."""

from concurrent import futures
from typing import Any, Callable


def run_in_parallel(
    function: Callable[..., Any],
    list_of_kwargs: list[dict[str, Any]],
    num_workers: int,
) -> list[Any]:
  """Runs a function in parallel with the given list of kwargs.

  Args:
    function: The function to run.
    list_of_kwargs: A list of kwargs, one for each invocation.
    num_workers: The number of parallel workers.

  Returns:
    A list of results, one for each invocation, in the same order.
  """
  if not list_of_kwargs:
    return []
  
  with futures.ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
    future_list = [
        executor.submit(function, **kwargs) for kwargs in list_of_kwargs
    ]
    return [f.result() for f in future_list]
