<<<<<<< HEAD
# Copyright (2025) critic-rl Authors 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
=======
# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
>>>>>>> 7010bb8 (update plot script)
# limitations under the License.
import argparse

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

<<<<<<< HEAD
=======
from ctrl.eval.code_utils import sanitize
>>>>>>> 7010bb8 (update plot script)
from ctrl.gen.prompt import get_prompter
from ctrl.gen.tree import NodeType


def cleanup_critique(critique: str):
    return critique.strip()


def main(args):
    # load dataset
    raw_df = pd.read_json(
        args.dataset,
        lines=True,
    )
    raw_df[args.id_field] = raw_df[args.id_field].astype(str)
    ds = raw_df.set_index(args.id_field).to_dict(orient="index")

    # track states for each task
    resume_df = pd.read_json(args.resume_file, lines=True)
    resume_df[args.id_field] = resume_df[args.id_field].astype(str)

    # extract previous data
    curr_df = resume_df[resume_df.node_type == NodeType.CRITIQUE_REVISION.value]
    prev_df = resume_df[resume_df.node_type == NodeType.GENERATION.value]

    # build critic instruction
<<<<<<< HEAD
    prompter = get_prompter(args.prompter_type)
=======
    prompter = get_prompter(args.dataset_type)
>>>>>>> 7010bb8 (update plot script)

    def build_critic_instruction(row):
        return prompter.get_critique_prompt(
            ds[row[args.id_field]][args.problem_field],
<<<<<<< HEAD
            prev_df[prev_df.hash == row.prev_hash]["sanitized_solution"].iloc[0],
=======
            sanitize(prev_df[prev_df.hash == row.prev_hash]).iloc[0],
>>>>>>> 7010bb8 (update plot script)
        )

    curr_df["critic_instruction"] = curr_df.apply(build_critic_instruction, axis=1)
    curr_df["critique"] = curr_df["critique"].apply(cleanup_critique)

    # filter out failed data
    if not args.keep_failed:
        curr_df = curr_df[curr_df[args.success_field] == 1]

    # filter out critiques that contain "the hint"
    curr_df = curr_df[~curr_df["critique"].str.contains("he hint")]

    # filter out prompts that are too long
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    curr_df = curr_df[
        curr_df["critic_instruction"].apply(lambda x: len(tokenizer.tokenize(x)))
        < args.max_prompt_len
    ]

    # balance the data
    if args.balance_data:
        prev_success_count = prev_df[prev_df[args.success_field] == 1].shape[0]
        prev_failed_count = prev_df[prev_df[args.success_field] == 0].shape[0]
        prev_ratio = prev_success_count / prev_failed_count
        curr_df["judgment"] = curr_df.critique.apply(
            lambda x: "Overall judgment: Correct" in x
        )
        judge_success_count = curr_df[curr_df["judgment"] == True].shape[0]
        judge_failed_count = curr_df[curr_df["judgment"] == False].shape[0]
        judge_ratio = judge_success_count / judge_failed_count

        # randomly drop data
        if judge_ratio > prev_ratio:
            # Need to drop successful cases
            target_success_count = int(prev_ratio * judge_failed_count)
            success_indices = curr_df[curr_df["judgment"] == True].index
            drop_indices = np.random.choice(
                success_indices,
                size=judge_success_count - target_success_count,
                replace=False,
            )
            curr_df = curr_df.drop(drop_indices)
        else:
            # Need to drop failed cases
            target_failed_count = int(judge_success_count / prev_ratio)
            failed_indices = curr_df[curr_df["judgment"] == False].index
            drop_indices = np.random.choice(
                failed_indices,
                size=judge_failed_count - target_failed_count,
                replace=False,
            )
            curr_df = curr_df.drop(drop_indices)

    # remove duplicates by id
    if args.remove_duplicates:
        curr_df = curr_df.drop_duplicates(subset=[args.id_field])

    curr_df = curr_df[["critic_instruction", "critique"]]
    curr_df.to_json(args.output_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_file", type=str, help="output file name")
    parser.add_argument(
        "--resume_file", type=str, default=None, help="resume file name"
    )

    # data parameters
    parser.add_argument("--id_field", type=str, default="task_id")
    parser.add_argument("--problem_field", type=str, default="problem")
    parser.add_argument("--success_field", type=str, default="success")
<<<<<<< HEAD
    parser.add_argument("--prompter_type", type=str, default="code_contests")
=======
    parser.add_argument("--dataset_type", type=str, default="code_contests")
>>>>>>> 7010bb8 (update plot script)
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--keep_failed", action="store_true")
    parser.add_argument("--balance_data", action="store_true")
    parser.add_argument("--remove_duplicates", action="store_true")

    # parse arguments
    args = parser.parse_args()

    main(args)
