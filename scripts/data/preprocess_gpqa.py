"""Adapted from https://github.com/huggingface/open-r1/blob/main/src/open_r1/evaluate.py"""

import json
import os
import random

import numpy as np
from datasets import load_dataset
from lighteval.tasks.requests import Doc


def gpqa_prompt_fn(line, task_name: str = None):
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""
    gold_index = random.randint(0, 3)
    choices = [
        line["Incorrect Answer 1"],
        line["Incorrect Answer 2"],
        line["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=line["Question"],
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )


if __name__ == "__main__":
    os.makedirs("scripts/data/gpqa", exist_ok=True)
    df = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train").to_pandas()

    def get_prompt_and_gt(row):
        doc = gpqa_prompt_fn(row)
        row['prompt'] = doc.query
        row['info'] = json.dumps(
                {
                    "ground_truth": doc.choices[doc.gold_index],
                }
            )
        return row

    df["task_id"] = np.arange(len(df))
    df["task_id"] = df.apply(lambda x: f"gpqa/test/{x['task_id']}", axis=1)
    df["dataset"] = "gpqa"
    df = df.apply(get_prompt_and_gt, axis=1)
    df[["task_id", "prompt", "dataset", "info"]].to_json("scripts/data/gpqa/test.jsonl", orient="records", lines=True)
