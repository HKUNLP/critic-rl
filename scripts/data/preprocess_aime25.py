"""Adapted from https://github.com/huggingface/open-r1/blob/main/src/open_r1/evaluate.py"""

import json
import os
import random

import numpy as np
from datasets import load_dataset
from lighteval.tasks.requests import Doc


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["answer"]],
        gold_index=0,
    )


if __name__ == "__main__":
    os.makedirs("scripts/data/aime25", exist_ok=True)
    df = load_dataset("yentinglin/aime_2025", split="train").to_pandas()

    def get_prompt_and_gt(row):
        doc = aime_prompt_fn(row)
        row["prompt"] = doc.query
        row["info"] = json.dumps(
            {
                "ground_truth": doc.choices[doc.gold_index],
            }
        )
        return row

    df["task_id"] = np.arange(len(df))
    df["task_id"] = df.apply(lambda x: f"aime25/test/{x['task_id']}", axis=1)
    df["dataset"] = "aime25"
    df = df.apply(get_prompt_and_gt, axis=1)
    df[["task_id", "prompt", "dataset", "info"]].to_json(
        "scripts/data/aime25/test.jsonl", orient="records", lines=True
    )
