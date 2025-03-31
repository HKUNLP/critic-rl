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
import base64
import json
import os
import pickle
import zlib

import pandas as pd
from datasets import load_dataset

io_format = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.
### Question:
{question}

### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)

"""

fn_format = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.
### Question:
{question}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```

### Answer: (use the provided format with backticks)

"""


def convert(row):
    # preprocess prompts
    if row.starter_code == "":  # io
        row["content"] = io_format.format(question=row.question_content)
    else:
        row["content"] = fn_format.format(
            question=row.question_content, starter_code=row.starter_code
        )

    # preprocess tests
    test = json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(row.private_test_cases)))
    )
    formatted_test = {"input_output": {"inputs": [], "outputs": [], "fn_name": None}}
    for i, t in enumerate(test):
        formatted_test["input_output"]["inputs"].append(t["input"])
        formatted_test["input_output"]["outputs"].append(t["output"])

    metadata = json.loads(row.metadata)
    if "func_name" in metadata:
        formatted_test["input_output"]["fn_name"] = metadata["func_name"]
    formatted_test["input_output"] = json.dumps(formatted_test["input_output"])
    row["test"] = json.dumps(formatted_test)

    row["labels"] = {
        "task_id": f"{row['platform']}/{row['question_id']}",
        "programming_language": "python",
        "execution_language": "python",
        "category": row["platform"],
        "difficulty": row["difficulty"],
        "fewshot": False,
    }
    row["labels"] = json.dumps(row["labels"])

    return row


if __name__ == "__main__":
    os.makedirs("scripts/data/livecodebench_20240811", exist_ok=True)
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag="release_v4",
        trust_remote_code=True,
    )

    for split in ds.keys():
        df = ds[split].to_pandas()
        df["contest_date"] = pd.to_datetime(df["contest_date"])
        mask = (df["contest_date"] >= "2024-08-01") & (
            df["contest_date"] <= "2024-11-01"
        )
        filtered_df = df[mask]

        filtered_df = filtered_df.apply(convert, axis=1)
        filtered_df["task_id"] = range(len(filtered_df))
        filtered_df["dataset"] = "livecodebench"
        filtered_df["prompt"] = filtered_df["content"]
        filtered_df = filtered_df[
            [
                "task_id",
                "prompt",
                "dataset",
                "content",
                "test",
                "labels",
            ]
        ]

        filtered_df["info"] = filtered_df.apply(
            lambda x: json.dumps(
                {
                    "id": x["task_id"],
                    "content": x["content"],
                    "test": x["test"],
                    "labels": x["labels"],
                }
            ),
            axis=1,
        )
        filtered_df["task_id"] = filtered_df.apply(
            lambda x: f"livecodebench_20240811/{split}/{x['task_id']}", axis=1
        )
        filtered_df[["prompt", "task_id", "info", "dataset"]].to_json(
            f"scripts/data/livecodebench_20240811/{split}.jsonl",
            lines=True,
            orient="records",
            force_ascii=False,
        )
