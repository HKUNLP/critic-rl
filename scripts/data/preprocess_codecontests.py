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

import numpy as np
from datasets import load_dataset


def convert_uts_to_sandbox(uts: dict) -> str:
    sandbox_tests = []
    for i, o in zip(uts["input"], uts["output"]):
        sandbox_test = {"input": {"stdin": i}, "output": {"stdout": o}}
        sandbox_tests.append(sandbox_test)
    return sandbox_tests


if __name__ == "__main__":
    os.makedirs("scripts/data/code_contests", exist_ok=True)
    ds = load_dataset("deepmind/code_contests")

    for split in ds.keys():
        df = ds[split].to_pandas()
        df["prompt"] = df["description"]
        df["public_uts"] = df["public_tests"].apply(convert_uts_to_sandbox)
        df["private_uts"] = df["private_tests"].apply(convert_uts_to_sandbox)
        df["generated_uts"] = df["generated_tests"].apply(convert_uts_to_sandbox)
        df["all_uts"] = df["public_uts"] + df["private_uts"] + df["generated_uts"]
        df["all_uts"] = df["all_uts"].apply(lambda x: x[:10])

        df["public_uts"] = df["public_uts"].apply(json.dumps)
        df["private_uts"] = df["private_uts"].apply(json.dumps)
        df["generated_uts"] = df["generated_uts"].apply(json.dumps)
        df["test"] = df["all_uts"].apply(json.dumps)
        df["task_id"] = np.arange(len(df))
        df["dataset"] = "code_contests"

        df = df[
            [
                "task_id",
                "prompt",
                "public_uts",
                "private_uts",
                "generated_uts",
                "test",
                "dataset",
            ]
        ]
        df["info"] = df.apply(
            lambda x: json.dumps(
                {
                    "test": x["test"],
                }
            ),
            axis=1,
        )
        df["task_id"] = df.apply(
            lambda x: f"codecontests/{split}/{x['task_id']}", axis=1
        )
        df[["prompt", "task_id", "info", "dataset"]].to_json(
            f"scripts/data/code_contests/{split}.jsonl",
            lines=True,
            orient="records",
            force_ascii=False,
        )
        del df
