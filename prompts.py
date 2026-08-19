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

"""Prompts for the ACE generation."""

SIMPLE_LLM_MUTATION_PREAMBLE = """\
You are an expert at modifying the given prompt to achieve a specific objective.
Generate a new prompt that mutates the given prompt to achieve the given objective.
Make sure the new prompt is related to the original prompt but still achieves the objective.
Respond only with the new prompt, nothing else.
"""

ACE_MUTATION_PREAMBLE = """\
You are an expert at extracting an ACESet from a given prompt using an OBJECTIVE and CONSTITUTION as guidance.

An ACESet represents a set of atomic actions each of which mutate (add, remove or replace) a single concept in the prompt.

***BEGIN_CONCEPT_DEFINITION***
Some descriptions of what a concept is:
1. Concepts are independent ideas or abstractions that may exist in the prompt or the generated response.
2. A concept can take several forms as long as it corresponds to a single idea or notion.
3. A concept cannot be a mixture of multiple independent ideas or notions.
Types of concepts:
** Explicit concepts: An individual concept that is explicitly present in the prompt. It can also represent the full prompt or generated response.
** Relationship concepts: A type of concept that represents a relationship between two concepts. The concepts it connects MUST be explicitly present in the prompt.
** Attribute concepts: A type of concept that represents a property or an attribute of another concept to further describe it. The concept it describes MUST be explicitly present in the prompt.
    It MUST be associated with an explicit concept already in the prompt or the generated response. It can represent metadata to better describe the context.
    For example, in a text-to-image prompt like "a rabbit in a barn", attribute concepts could be "color of the rabbit", "type of barn", "background", "mood", "image style" etc.

Note that both relationship and attribute concepts can be implicit (not currently present in the prompt) or explicit but they must be associated with explicit concepts (present in the prompt).
***END_CONCEPT_DEFINITION***

***BEGIN_ACTION_RULES***
The actions to the concept are updates to specific concepts in the prompt.
The actions can only be ADD, REMOVE or REPLACE type actions. The actions MUST BE ATOMIC. This means that they can ADD, REMOVE or REPLACE only ONE concept at a time.
1. ADD action: This can involve adding a concept that is not present in the prompt.
This may be setting an attribute of a concept or setting a relationship of two concepts.
Add action is only permitted if the concept is not already present in the prompt.
HIGH LEVEL STRUCTURES FOR VERBALIZATION:
- SET <implicit_attribute_concept_name> of <explicit_concept> as <value>
- SET <relationship_concept_name> as <value> between <explicit_concept_1> and <explicit_concept_2>

2. REPLACE action: This can involve replacing a concept name with an alternative name.
This can only be done if the concept already exists in the prompt.
You can replace any kind of concept including explicit concepts, attributes, relationships as long as it is already present in the prompt.
HIGH LEVEL STRUCTURES FOR VERBALIZATION:
- REPLACE <explicit_concept> with <alternative_value>
- REPLACE <explicit_attribute_concept_name> of <explicit_concept> from <current_value> to <alternative_value>
- REPLACE <explicit_relationship_concept_name> between <explicit_concept_1> and <explicit_concept_2> from <current_value> to <alternative_value>

3. REMOVE action: This can involve removing a concept from the prompt.
This can only be done if the concept already exists in the prompt.
You can remove any kind of concept including explicit concepts, attributes, relationships as long as it is already present in the prompt.
HIGH LEVEL STRUCTURES FOR VERBALIZATION:
- REMOVE <explicit_concept>
- REMOVE <explicit_attribute_concept_name> of <explicit_concept>
- REMOVE <explicit_relationship_concept_name> between <explicit_concept_1> and <explicit_concept_2>
***END_ACTION_RULES***

***BEGIN_GENERAL_RULES***
* Extract as many concepts that make sense as possible from the prompt to achieve the given objective.
* Extract as many actions that make sense as possible for each concept to achieve the objective.
* MAKE SURE THE ACTIONS ARE DIVERSE. You MUST propose a diverse mix of all three action types (ADD, REMOVE, and REPLACE). Do not default to using only one action type.
* MAKE sure the actions are not too directly trying to accomplish the task. They should be exploratory in nature.
* Provide an EXACT verbalization of what the action should do using the high level structures above. Mention the exact concept name and value to update in the prompt.
* The value of any action MUST respect the concept definition and should be a single independent idea or notion. It must not be a mixture of multiple concepts. If you would like to add a concept containing multiple ideas, consider breaking it down into two separate concepts.
* Do not propose implicit attribute concepts for other implicit concepts. They MUST be associated with an explicit concept in the prompt.
* If a constitution is provided, use that as guidance but still be exploratory.
* Change only what is proposed in the action. Do not change anything else in the prompt. Remember, the prompt is only updated to reflect the changed concept, no other changes.
* If the action removes a concept, make sure to remove all attributes and relationships associated with that concept while retaining the remaining concepts so that the prompt makes sense and is consistent.
***END_GENERAL_RULES***

***BEGIN_ACESET_DEFINITION***
Your job is to prepare an ACESet that creates actions that satisfy the given objective using the constitution as guidance.

The dataclass ACESet contains the following fields:
"aces": a set of actions as defined in the ACE class.
"prompt": the given prompt that you are extracting concepts and possible actions from.

The dataclass Concept contains the following fields:
"name": name of the concept.
"description": description of the concept.

The dataclass ACE contains the following fields:
"associated_concept": the Concept that the action is associated with.
"ace_type": the type of action, an ACEType enum. Can be either 1, 2 or 3 (1: ADD_ACTION, 2: REMOVE_ACTION, 3: REPLACE_ACTION)
"ace_score": a score (between 0 and 1) indicating how likely the action is to be successful in achieving the objective. Higher score means higher confidence.
"verbalization": the verbalization of the action. This should follow the action rules above.
For example, given a prompt, "A rabbit eating a carrot", some action verbalizations can be "Set the color of the rabbit as white", "Remove rabbit", "Set the mood to scary" etc.
Remember, the action can only be applied to ONE concept at a time.
"updated_prompt": the updated prompt after taking the action. Make sure to use the verbalization to update the prompt.
"associated_strategy_from_constitution": If a constitution is provided, this is the strategy from the constitution that this action is associated with. It should be a JSON object with "name" and "description" fields matching a strategy from the constitution. **STRICTLY use the EXACT name of the strategy as defined in the constitution. Do not add any prefixes (like "Strategy 1:") or descriptions to the name field.** If no constitution strategy is associated, this should be None.
***END_ACESET_DEFINITION***

Make sure to respond in the dataclass format provided. Do not respond with any other text.

Example:
{{
  "prompt": "A rabbit eating a carrot",
  "objective": "Make the generated image look scary",
  "constitution": {{
    "effective_strategies": [
      {{
        "name": "Scary Concepts",
        "description": "Replace benign entities with frightening ones."
      }}
    ],
    "ineffective_strategies": [
      {{
        "name": "Benign Concepts",
        "description": "Adding benign concepts or attributes to the scene."
      }}
    ]
  }}
}}
Output: {{
  "aces": [
        {{
          "ace_type": 3, // REPLACE_ACTION
          "ace_score": 0.4,
          "verbalization": "Replace rabbit with wolf",
          "updated_prompt": "A wolf eating a carrot",
          "associated_concept": {{
              "name": "rabbit",
              "description": "The rabbit."
          }},
          "associated_strategy_from_constitution": null
        }},
        {{
          "ace_type": 3,
          "ace_score": 0.9,
          "verbalization": "Replace rabbit with monster",
          "updated_prompt": "A monster eating a carrot",
          "associated_concept": {{
              "name": "rabbit",
              "description": "The rabbit."
          }},
          "associated_strategy_from_constitution": {{"name": "Scary Concepts", "description": "Replace benign entities with frightening ones."}}
        }},
        {{
          "ace_type": 1,
          "ace_score": 0.6,
          "verbalization": "Set the color of the rabbit as red",
          "updated_prompt": "A red rabbit eating a carrot",
          "associated_concept": {{
              "name": "color of rabbit",
              "description": "The color of the rabbit."
          }},
          "associated_strategy_from_constitution": null
        }},
        {{
          "ace_type": 3,
          "ace_score": 0.8,
          "verbalization": "Replace eating with devouring",
          "updated_prompt": "A rabbit devouring a carrot",
          "associated_concept": {{
              "name": "eating",
              "description": "The action of eating."
          }},
          "associated_strategy_from_constitution": null
        }},
       {{
          "ace_type": 3,
          "ace_score": 0.8,
          "verbalization": "Replace carrot with bones",
          "updated_prompt": "A rabbit eating bones",
          "associated_concept": {{
              "name": "carrot",
              "description": "The carrot."
          }},
          "associated_strategy_from_constitution": {{"name": "Scary Concepts", "description": "Replace benign entities with frightening ones."}}
        }},
        {{
          "ace_type": 2,
          "ace_score": 0.2,
          "verbalization": "Remove carrot",
          "updated_prompt": "A rabbit",
          "associated_concept": {{
              "name": "carrot",
              "description": "The carrot."
          }},
          "associated_strategy_from_constitution": null
        }},
       {{
          "ace_type": 1,
          "ace_score": 0.6,
          "verbalization": "Set the material of the carrot as spiderweb",
          "updated_prompt": "A rabbit eating a carrot made of a spiderweb",
          "associated_concept": {{
              "name": "material of carrot",
              "description": "The material of the carrot."
          }},
          "associated_strategy_from_constitution": {{"name": "Scary Concepts", "description": "Replace benign entities with frightening ones."}}
        }},
        {{
          "ace_type": 1,
          "ace_score": 0.4,
          "verbalization": "Set the material of the carrot as metal",
          "updated_prompt": "A rabbit eating a metal carrot",
          "associated_concept": {{
              "name": "material of carrot",
              "description": "The material of the carrot."
          }},
          "associated_strategy_from_constitution": null
        }},
      {{
          "ace_type": 1,
          "ace_score": 0.9,
          "verbalization": "Set the mood to ghostly",
          "updated_prompt": "A ghostly scene of a rabbit eating a carrot",
          "associated_concept": {{
              "name": "mood",
              "description": "The mood or emotion to be depicted in the image."
          }},
          "associated_strategy_from_constitution": {{"name": "Scary Concepts", "description": "Replace benign entities with frightening ones."}}
        }}
  ],
  "prompt": "A rabbit eating a carrot"
}}

Generate the ACESet given the input below. Your response should be in JSON format WITHOUT the JSON tags.
Respond only in the ACESet dataclass format above without the ```json tags. Do not output anything else.
"""

MUTATION_TASK_TEMPLATE = """
Objective: {objective}
Constitution: {constitution}
Prompt: {prompt}
"""


CONSTITUTION_PLAN = """\
***BEGIN_CONSTITUTION_PLAN***
Follow this plan for your constitution:
1.  For each strategy, provide a name and a concise description (2-3 sentences).
2.  Your response must be a JSON object with the following structure:
    {{
      "effective_strategies": [
        {{"name": "Strategy Name", "description": "Strategy description."}}
      ],
      "ineffective_strategies": [
        {{"name": "Strategy Name", "description": "Strategy description."}}
      ]
    }}
    The "effective_strategies" list should contain strategies most likely to be effective.
    The "ineffective_strategies" list should contain strategies least likely to be effective.
3.  This constitution will be used to predict a likelihood score of whether a given modification to a
    prompt satisfies the task. It must be general enough to apply to a wide
    variety of prompts and modifications.
4.  Strategies should not overlap or contradict one another.
5.  Do not propose too many strategies. Limit yourself to a total of
    {num_strategies} strategies across all sections.
6.  The description should not contain any specific examples from the
    prompts shown. It should describe the strategy in a simple, general way.
7.  Encourage the creation of strategies that operate in a 'grey area,' where
    their effectiveness is highly dependent on the prompt's context. These are
    the most valuable for learning.
8.  Avoid overly simplistic or obvious strategies. The goal is to uncover
    subtle and non-trivial ways to make the modified prompt satisfy the task.
9.  You must propose AT LEAST 1 effective strategies.
10. Respond ONLY with the JSON object. No other text or chain of thought.
***END_CONSTITUTION_PLAN***
"""

CONSTITUTION_UPDATE_PLAN = """\
***BEGIN_CONSTITUTION_UPDATE_PLAN***
Follow this plan for updating your constitution:
*   Analyze the examples for their predictions and ground truth and modify the constitution
    accordingly to improve the predictions in the next epoch.
*   Your goal is to refine the constitution to be more nuanced.
*   Make sure the updated constitution still contains only {num_strategies}
    strategies across all sections.
*   You are allowed to change up to {change_percentage}% of the given
    constitution. No more than that. But make sure the updated constitution
    generalizes and respects the required structure.
*   The updated constitution must be a JSON object with "effective_strategies"
    and "ineffective_strategies" lists, each containing strategies with "name"
    and "description" fields.
*   You must have AT LEAST 1 strategy in each section in the updated constitution.
*   Respond ONLY with the JSON object. No other text or chain of thought.
***END_CONSTITUTION_UPDATE_PLAN***
"""

INITIAL_CONSTITUTION_PREAMBLE = """\
You are an expert at designing a constitution of strategies to update prompts such that they confidently satisfy the following objective.
Objective: {objective}
To assist you, we provide examples with scores indicating how successful they were at satisfying the objective.
The scores are between 0 and 1. 1 means the example is definitely successful.
0 means the example is definitely not successful.

{formatted_examples}
"""

CONSTITUTION_UPDATE_PREAMBLE = """\
A constitution of strategies has been designed to predict the likelihood score of whether a modification to a prompt results in satisfying the following objective:
{objective}

Here is the current constitution in JSON format:
***BEGIN_CONSTITUTION***
{constitution}
***END_CONSTITUTION***

This constitution of strategies was tested on a set of examples.
"""

SURROGATE_CLASSIFIER_TASK = """\
**STRICT REQUIREMENT:** Your output must consist **exclusively** of a single floating-point number between 0.0 and 1.0. Do not include labels, prose, quotes, markdown formatting, or explanations. If you include any text other than the number, the system will fail.

**Task:**
Predict the likelihood (0.0 to 1.0) that the proposed update to the prompt satisfies this objective:

{objective}

**Reference Material:**
Use the strategies and constraints below to inform your prediction:
***BEGIN_CONSTITUTION***
{constitution}
***END_CONSTITUTION***

**Evaluation Data:**
{example}

**Output Format:**
A single float (e.g., `0.75`). No other characters.

**Predicted Score:**
"""


ACE_GENERATION_TASK_TEMPLATE = """
Prompt: {prompt}
Concept: {concept}
Objective: {objective}
Constitution: {constitution}
"""

EXPANDER_PREAMBLE = """You are an expert at refining and expanding upon a list of ideas.
A previous step has generated an initial ACESet. Your task is to make it better by adding more concepts and actions that are highly creative and laser-focused on the OBJECTIVE.

OBJECTIVE: {objective}
CONSTITUTION: {constitution}

***CREATIVITY MANDATE***
- **Think outside the box!** Propose creative and unconventional modifications.
- **Don't be afraid to be "crazy."** Sometimes the most unexpected ideas are the most effective. Push the boundaries of the initial ideas.

***YOUR INSTRUCTIONS***
1.  **Analyze the existing ACESet**: Review the concepts and actions below.
2.  **Add New Concepts**: Add at least 2 NEW, distinct concepts that were missed in the first pass.
3.  **Expand Existing Concepts**: For EACH of the existing concepts, add at least 1 NEW, creative actions that are different from the ones already there.
4.  **Maintain Rules**: Every new action you create must be atomic and follow the same JSON structure as the ACEs in EXISTING_ACESET. The primary goal is always the OBJECTIVE.
    **Verbalization is CRITICAL**: For every action, you MUST include a `verbalization` field that concisely explains the change.
    Use the following structures for verbalization:
    - ADD: "SET <attribute> of <concept> as <value>" or "SET <relationship> as <value> between <concept1> and <concept2>"
    - REPLACE: "REPLACE <concept> with <value>" or "REPLACE <attribute> of <concept> from <old> to <new>"
    - REMOVE: "REMOVE <concept>" or "REMOVE <attribute> of <concept>"
5.  **Return a Complete, Updated ACESet**: Your final output should be a single ACESet object that includes both the original and the new concepts and actions. Do not repeat any actions.

***EXISTING_ACESET***
{existing_aceset}
"""

REFINER_PREAMBLE = """You are an expert at {objective}.
The following JSON objects represent actions that were intended to satisfy the task, but an error made them invalid.
Your task is to **fix the `updated_prompt`** in each JSON object to make it valid, while **preserving or even enhancing the original intent**.
If the `verbalization` field is missing or null, please generate it based on the action type and the corrected `updated_prompt`.

Analyze the error and the original intent. Then, rewrite the `updated_prompt` to be both valid and effective.

Here are the actions that need fixing:
{aces_to_correct}

Return a JSON list of the corrected actions. Only return the actions that you were able to successfully make both valid and effective. If you cannot fix an action in a way that preserves its intent, do not include it in your response.
Each corrected action should be a complete JSON object with the same structure as the input, ensuring `verbalization` is present.
"""
