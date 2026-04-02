# Atomic Concept Edits (ACE)

ACE is a framework for systematic prompt exploration and optimization through
atomic concept-level edits. It decomposes prompts into individual concepts
and generates targeted mutations (add, remove, replace) to explore how small
changes affect model behavior with respect to a given objective. The framework
includes:

- **ACE Exploration**: A tree-search algorithm that iteratively
  mutates prompts, evaluates them against a scorer that measures
  the objective (generally including a target model and autorater),
  and expands the most promising branches.
- **Constitution Optimizer**: An optimization loop that learns a
  "constitution" (a set of strategies) to guide the mutation
  process, using a surrogate classifier to predict which edits
  will satisfy the objective. This constitution uncovers key
  aspects of the model's behavior with respect to the objective
  and can then be used to guide the ACE exploration.

## Installation

The project uses `pyproject.toml` for package configuration.
To install it in editable mode:

```bash
python -m venv ace_env
source ace_env/bin/activate
pip install -e .
```

To install with all optional dependencies
(OpenAI, Anthropic, Hugging Face Datasets, and TensorFlow Datasets):

```bash
pip install -e ".[all]"
```

## Usage

### ACE Exploration

Simple demo: ace_word_count_demo.ipynb

You can run ACE exploration with a config file:

```bash
python run_ace_exploration.py --config=configs/word_count.py
```

### Constitution Optimizer

After running an exploration, you can optimize a constitution:

```bash
python run_constitution_optimizer.py --exploration_output_dir='<path_to_exploration_output_dir>'
```

## Citing this work

```
@article{kalibhat2026interpretingcontrollingmodelbehavior,
      title={Interpreting and Controlling Model Behavior via Constitutions for Atomic Concept Edits},
      author={Neha Kalibhat and Zi Wang and Prasoon Bajpai and Drew Proud and Wenjun Zeng and Been Kim and Mani Malek},
      year={2026},
      eprint={2602.00092},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.00092},
}
```

## License and disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
