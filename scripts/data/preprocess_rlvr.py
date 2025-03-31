import json
import os

import numpy as np
from datasets import load_dataset


if __name__ == "__main__":
    os.makedirs("scripts/data/rlvr", exist_ok=True)
    ds = load_dataset("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints")

    for split in ds:
        df = ds[split].to_pandas()
        df["prompt"] = df["messages"].apply(lambda x: x[0]["content"])
        df["task_id"] = np.arange(len(df))

        df["info"] = df.apply(
            lambda row: json.dumps(
                {
                    "ground_truth": row["ground_truth"],
                }
            ),
            axis=1,
        )

        df["task_id"] = df.apply(lambda x: f"rlvr/{split}/{x['task_id']}", axis=1)
        df[["prompt", "task_id", "info", "dataset"]].to_json(
            f"scripts/data/rlvr/{split}.jsonl", orient="records", lines=True
        )
