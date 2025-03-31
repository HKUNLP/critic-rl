import json
import os

import numpy as np
from datasets import load_dataset


def format_prompt(prompt):
    new_prompt = prompt
    new_prompt = new_prompt.split("TL;DR:")[0].strip()
    new_prompt = f"""You are an expert at summarizing text. Please write a TL;DR of the following post:

<post>
{new_prompt}
</post>

Make sure to only include the TL;DR and nothing else in your response."""
    return new_prompt


if __name__ == "__main__":
    os.makedirs("scripts/data/tldr", exist_ok=True)
    ds = load_dataset("CarperAI/openai_summarize_tldr")

    for split in ds:
        df = ds[split].to_pandas()
        df["prompt"] = df["prompt"].apply(format_prompt)
        df["task_id"] = np.arange(len(df))
        df["dataset"] = "tldr"

        df["info"] = df.apply(
            lambda row: json.dumps(
                {
                    "ground_truth": row["label"],
                }
            ),
            axis=1,
        )

        df["task_id"] = df.apply(lambda x: f"tldr/{split}/{x['task_id']}", axis=1)
        df[["prompt", "task_id", "info", "dataset"]].to_json(
            f"scripts/data/tldr/{split}.jsonl", orient="records", lines=True
        )
