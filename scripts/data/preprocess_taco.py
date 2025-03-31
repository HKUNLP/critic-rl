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
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

sys.set_int_max_str_digits(100000)


def convert_uts_to_sandbox(row: pd.Series) -> str:
    uts = row["input_output"]
    sandbox_tests = []

    for i, o in zip(uts["inputs"], uts["outputs"]):
        if "fn_name" not in uts and (isinstance(i, list) or isinstance(o, list)):
            continue

        if "fn_name" not in uts and (i.strip() == ""):
            continue

        sandbox_test = {"input": {"stdin": i}, "output": {"stdout": o}}
        sandbox_tests.append(sandbox_test)
    return sandbox_tests


def add_fn_info(row):
    fn_name = row["fn_name"]
    if fn_name is None:
        return row

    if f"{fn_name}(" not in row["prompt"]:
        row["prompt"] = (
            f"{row['prompt']}\n\nYour code should satisfy these tests:\n```python\n{row['all_uts'][0]}\n```"
        )

    return row


def process_input_output(inputs, outputs):
    """From https://github.com/FlagOpen/TACO"""
    # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
    try:
        if isinstance(inputs[0], dict):
            inputs = [{int(k): v for k, v in inputs[0].items()}]
    except:
        True

    try:
        if isinstance(outputs, dict):
            outputs = [{int(k): v for k, v in outputs.items()}]
    except:
        True

    try:
        if isinstance(outputs[0], dict):
            outputs = [{int(k): v for k, v in outputs[0].items()}]
    except:
        True

    # Simple fix for list outputs
    if isinstance(outputs, list) and outputs:
        outputs = outputs[0]

    return inputs, outputs


def separate_io_fncall(row):
    io = row["input_output"]

    if io.get("fn_name"):
        row["fn_name"] = io["fn_name"]
    else:
        row["fn_name"] = None

    return row


def create_function_call_str(func_name, args_list):
    args_str = ", ".join(repr(arg) for arg in args_list)
    return f"{func_name}({args_str})"


def format_uts(row):
    io = row["input_output"]

    uts_list = []

    for inputs, outputs in zip(io["inputs"], io["outputs"]):
        inputs, outputs = process_input_output(inputs, outputs)

        if row["fn_name"] is None:
            if " = " in inputs or "[]" in inputs:
                continue
            if isinstance(inputs, list):
                continue

            cc_formatted_ut = {
                "input": {"stdin": inputs},
                "output": {"stdout": outputs},
            }
            uts_list.append(cc_formatted_ut)
        else:
            if isinstance(inputs, str):
                continue

            runcode_formatted_ut = f"assert {create_function_call_str(row['fn_name'], inputs)} == {repr(outputs)}".replace(
                "'\"", '"'
            ).replace(
                "\"'", '"'
            )
            uts_list.append(runcode_formatted_ut)

    row["all_uts"] = uts_list
    return row


if __name__ == "__main__":
    os.makedirs("scripts/data/taco", exist_ok=True)
    ds = load_dataset("BAAI/TACO", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    for split in ds.keys():
        df = ds[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["prompt"] = df["question"].apply(lambda x: x.strip())
        df["input_output"] = df["input_output"].apply(json.loads)
        df = df.apply(separate_io_fncall, axis=1)
        df = df.apply(format_uts, axis=1)

        # filter out single uts
        df = df[df.all_uts.apply(lambda x: len(x) > 0)]

        # filter out <image> prompt
        df = df[df.prompt.apply(lambda x: "<image>" not in x)]

        # filter out prompt that contains weird characters
        df = df[df.prompt.apply(lambda x: "<span " not in x)]

        # control prompt length
        if split == "train":
            df["prompt_len"] = df["prompt"].apply(lambda x: len(tokenizer.tokenize(x)))
            df = df[df["prompt_len"] < 2048]
            df = df[df["prompt_len"] > 64]

        df["all_uts"] = df.apply(
            lambda row: (
                json.dumps(row["all_uts"], ensure_ascii=False)
                if row["fn_name"] is None
                else row["all_uts"]
            ),
            axis=1,
        )

        df = df.apply(add_fn_info, axis=1)

        # extract timelimit
        def extract_float1(string):
            if string is None:
                return None
            match = re.search(r"[-+]?\d*\.\d+|\d+", string)
            return float(match.group()) if match else None

        df["time_limit"] = df["time_limit"].apply(extract_float1)

        # deduplicate
        df = df.drop_duplicates(subset=["prompt"])

        # remove prompts that are in code_contests
        cc_prompts = load_dataset("deepmind/code_contests", split="test")["description"]
        df = df[~df["prompt"].isin(cc_prompts)]

        # rename
        df.rename(columns={"all_uts": "test"}, inplace=True)

        # add dataset label
        df["dataset"] = df["fn_name"].apply(
            lambda x: "mbppplus" if x is not None else "code_contests"
        )

        # add info label
        df["info"] = df.apply(
            lambda row: json.dumps(
                {
                    "test": row["test"],
                    "fn_name": row["fn_name"],
                    "time_limit": row["time_limit"],
                }
            ),
            axis=1,
        )

        # save to jsonl
        df["task_id"] = df.apply(lambda x: f"taco/{split}/{x['task_id']}", axis=1)
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/taco/{split}.jsonl",
            lines=True,
            orient="records",
            force_ascii=False,
        )
        del df
