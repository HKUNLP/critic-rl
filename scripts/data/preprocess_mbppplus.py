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


def format_problem(prompt, test):
    return f"{prompt.strip()} Your code should satisfy these tests:\n\n{test.strip()}"


if __name__ == "__main__":
    os.makedirs("scripts/data/mbppplus", exist_ok=True)
    ds = load_dataset("evalplus/mbppplus")

    for split in ds:
        df = ds[split].to_pandas()
        df["prompt"] = df.apply(
            lambda x: format_problem(x["prompt"], x["test_list"][0]), axis=1
        )
        df["task_id"] = np.arange(len(df))
        df["dataset"] = "mbppplus"
        df["info"] = df.apply(
            lambda x: json.dumps(
                {
                    "test": x["test"],
                }
            ),
            axis=1,
        )

        df["task_id"] = df.apply(lambda x: f"mbppplus/{split}/{x['task_id']}", axis=1)
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/mbppplus/{split}.jsonl",
            lines=True,
            orient="records",
            force_ascii=False,
        )
