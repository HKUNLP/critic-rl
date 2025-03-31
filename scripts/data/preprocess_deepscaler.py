import json
import os

import numpy as np
from datasets import load_dataset


if __name__ == "__main__":
    os.makedirs("scripts/data/deepscaler", exist_ok=True)
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(lambda x: f"deepscaler/{split}/{x['task_id']}", axis=1)
        df["dataset"] = "deepscaler"
        df["prompt"] = df.problem
        df["info"] = df.apply(
            lambda x: json.dumps({"ground_truth": x["answer"]}), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepscaler/{split}.jsonl", orient="records", lines=True
        )
