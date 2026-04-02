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

r"""Run ACE exploration.

Example usage:
python run_ace_exploration.py \
--config=configs/word_count.py
--config.ace_exploration_config.run_id=<optional_run_id_to_resume>
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from atomic_concept_edits import ace_exploration
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', '', 'Configuration file.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = FLAGS.config
  exploration = ace_exploration.ACEExploration(config.ace_exploration_config)
  exploration.run()


if __name__ == '__main__':
  app.run(main)
