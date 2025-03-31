# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A function reward model for CodeContests (critic training)
"""
import ast
import asyncio
import json
import re
from typing import Any, List

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizer
from verl import DataProto

from ctrl.gen.prompt import extract_critique_from_xml, get_prompter
from ctrl.eval.score_utils import apply_verifiable_reward

RUN_TIMEOUT = 10
MAX_REQUESTS = 256


def normalize_code(code_str):
    try:
        # Parse the code into an AST
        tree = ast.parse(code_str)

        # Dictionary to store variable name mappings
        var_counter = 0
        var_map = {}

        # AST transformer to rename variables
        class VariableRenamer(ast.NodeTransformer):
            def visit_Name(self, node):
                nonlocal var_counter
                if isinstance(node.ctx, ast.Store):
                    if node.id not in var_map:
                        var_map[node.id] = f"v_{var_counter}"
                        var_counter += 1
                return ast.Name(id=var_map.get(node.id, node.id), ctx=node.ctx)

        # Apply the transformation
        transformed = VariableRenamer().visit(tree)

        # Convert back to string with normalized formatting
        normalized = ast.unparse(transformed)
        return normalized

    except SyntaxError:
        # If parsing fails, return original code
        return code_str


class RewardModelForCritic(object):
    def __init__(
        self,
        file,  # train_file or test_file
        tokenizer: PreTrainedTokenizer,
        num_examine: int = 0,
        run_all_cases: bool = True,
    ):
        self.tokenizer = tokenizer

        # load train file or test file
        self.file = file

        self.num_examine = num_examine

        if isinstance(self.file, str):
            self.dataset = pd.read_parquet(self.file)
        else:
            self.dataset = pd.concat([pd.read_parquet(f) for f in self.file])

        self.dataset["proxy_id"] = self.dataset["task_id"]
        self.id_to_infos = self.dataset.set_index("proxy_id").to_dict(orient="index")
        self.run_all_cases = run_all_cases

    def parse_query(self, ids: List[str]) -> List[pd.Series]:
        init_messages, rows = [], []

        for id in ids:
            row = self.id_to_infos[id]
            prompter = get_prompter(row["dataset"])

            init_message = [
                {
                    "role": "user",
                    "content": prompter.get_gen_prompt(row["gen_prompt"]),
                },
                {
                    "role": "assistant",
                    "content": row["solution"],
                },
            ]

            init_messages.append(init_message)
            rows.append(row)
        return init_messages, rows

    def build_revision_messages(self, init_messages, critique):
        # generate revision
        assert isinstance(init_messages, list)
        assert init_messages[-1]["role"] == "assistant"

        revision_prompt = f"""{extract_critique_from_xml(critique).strip()}

Please finalize your answer accordingly using the same format."""

        return init_messages + [{"role": "user", "content": revision_prompt}]

    def parse_response(self, data: DataProto) -> DataProto:
        ids = []
        generations = []
        valid_response_lengths = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(valid_response_ids)
            # remove <eos> token
            response_str = response_str.replace(self.tokenizer.eos_token, "")

            if i < self.num_examine:
                print(response_str)

            # get test case
            task_id = data_item.non_tensor_batch["task_id"]
            ids.append(task_id)
            generations.append(response_str)
            valid_response_lengths.append(valid_response_length)

        return ids, generations, valid_response_lengths

    def _apply_format_reward(self, critique: str):
        pattern = r"""Analysis:\n\s*(.+?)\s*
\n\nImprovement\s+suggestions:\n\s*(.+?)\s*
\n\nOverall\s+judgment:\s*(Correct|Incorrect)$"""

        match = re.search(pattern, critique, re.DOTALL | re.VERBOSE)

        if not match:
            return False

        analysis, suggestions, judgment = match.groups()
        return float(
            bool(analysis.strip() and suggestions.strip() and judgment.strip())
        )

    async def reward_revision(
        self, critique: str, revision: str, sample: dict, semaphore: asyncio.Semaphore
    ) -> float:
        format_reward = self._apply_format_reward(critique)

        if not format_reward:
            return -1.0

        # verifiable reward
        dataset, info = sample["dataset"], json.loads(sample["info"])
        reward, _ = await apply_verifiable_reward(revision, dataset, info)

        return reward

    async def get_reward_all(
        self, critiques: List[str], revisions: List[str], samples: List[Any]
    ) -> List[float]:
        semaphore = asyncio.Semaphore(MAX_REQUESTS)
        rewards = await tqdm_asyncio.gather(
            *[
                self.reward_revision(critique, revision, sample, semaphore)
                for critique, revision, sample in zip(critiques, revisions, samples)
            ],
            desc="Generating rewards",
        )
        return rewards
