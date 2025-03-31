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
"""Convert data to prompt data for verl"""

import argparse

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ctrl.gen.prompt import get_prompter


def main(args):
    df = pd.read_json(args.input, lines=True)

    # format context messages
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def format_prompt(row):
        prompter = get_prompter(row["dataset"])
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": prompter.get_gen_prompt(row["prompt"]),
            }
        )
        if args.keep_chat:
            return messages
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    df["prompt"] = df.apply(format_prompt, axis=1)

    df["task_id"] = (
        df["task_id"] + "/" + df.groupby("task_id").cumcount().astype(str)
    )  # avoid duplicate task_id
    df.rename(columns={'dataset': 'data_source'}, inplace=True)
    df.to_parquet(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file name")
    parser.add_argument("output", type=str, help="output file name")

    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    )
    parser.add_argument("--keep_chat", action="store_true")
    args = parser.parse_args()

    main(args)
