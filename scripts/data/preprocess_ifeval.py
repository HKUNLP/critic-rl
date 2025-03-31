import json
import os

import numpy as np
from datasets import load_dataset


def format_prompt(prompt):
    new_prompt = prompt
    new_prompt = f"""You are an assistant that carefully follows instructions. Here is an instruction that describes a task.

<instruction>
{new_prompt}
</instruction>

Make sure to follow the instruction exactly and not add any additional information in your response."""
    return new_prompt


if __name__ == "__main__":
    os.makedirs("scripts/data/ifeval_ori", exist_ok=True)
    ds = load_dataset("google/IFEval")

    for split in ds:
        df = ds[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["dataset"] = "ifeval_ori"
        df["info"] = df.apply(
            lambda row: {
                "key": row["key"],
                "prompt": row["prompt"],
                "kwargs": list(row["kwargs"]),
                "instruction_id_list": list(row["instruction_id_list"]),
            },
            axis=1,
        )
        df["prompt"] = df["prompt"].apply(format_prompt)
        df["task_id"] = df.apply(lambda x: f"ifeval_ori/{split}/{x['task_id']}", axis=1)
        df[["prompt", "task_id", "info", "dataset"]].to_json(
            f"scripts/data/ifeval_ori/{split}.jsonl", orient="records", lines=True
        )
