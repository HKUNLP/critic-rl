import json
import os

import numpy as np
from datasets import load_dataset


def create_function_call_str(func_name, args_list):
    args_str = ", ".join(repr(arg) for arg in args_list)
    return f"{func_name}({args_str})"


def preprocess_uts(row):
    tests = row["tests"]
    tests = json.loads(tests)
    new_tests = []
    dataset = "undefined"

    for t in tests:
        if "fn_name" in t:
            runcode_formatted_ut = f"assert {create_function_call_str(t['fn_name'], t['input'])} == {repr(t['output'][0])}".replace(
                "'\"", '"'
            ).replace(
                "\"'", '"'
            )
            new_tests.append(runcode_formatted_ut)
            dataset = "mbppplus"
        else:
            t["input"] = {"stdin": t["input"]}
            t["output"] = {"stdout": t["output"]}
            new_tests.append(t)
            dataset = "code_contests"

    row["info"] = json.dumps({"test": json.dumps(new_tests)})
    row["dataset"] = dataset
    return row


def preprocess_uts_taco(row):
    tests = row["tests"]
    tests = json.loads(tests)
    new_tests = []
    dataset = "code_contests" if "fn_name" not in tests else "mbppplus"

    for i, o in zip(tests["inputs"], tests["outputs"]):
        if dataset == "code_contests":
            t = {}
            t["input"] = {"stdin": i}
            t["output"] = {"stdout": o}
            new_tests.append(t)
        else:
            runcode_formatted_ut = f"assert {create_function_call_str(tests['fn_name'], i)} == {eval(repr(o))[0]}".replace(
                "'\"", '"'
            ).replace(
                "\"'", '"'
            )
            new_tests.append(runcode_formatted_ut)

    row["info"] = json.dumps({"test": json.dumps(new_tests)})
    row["dataset"] = dataset
    return row


if __name__ == "__main__":
    os.makedirs("scripts/data/deepcoder", exist_ok=True)
    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "codeforces")

    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(
            lambda x: f"deepscaler-codeforces/{split}/{x['task_id']}", axis=1
        )
        df["prompt"] = df.problem
        df = df.apply(
            lambda x: preprocess_uts(x), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/codeforces_{split}.jsonl",
            orient="records",
            lines=True,
        )

    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5")

    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(
            lambda x: f"deepscaler-lcbv5/{split}/{x['task_id']}", axis=1
        )
        df["prompt"] = df.problem
        df = df.apply(
            lambda x: preprocess_uts(x), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/lcbv5_{split}.jsonl",
            orient="records",
            lines=True,
        )

    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect")

    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(
            lambda x: f"deepscaler-primeintellect/{split}/{x['task_id']}", axis=1
        )
        df["prompt"] = df.problem
        df = df.apply(
            lambda x: preprocess_uts(x), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/primeintellect_{split}.jsonl",
            orient="records",
            lines=True,
        )

    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "taco")

    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(
            lambda x: f"deepscaler-taco/{split}/{x['task_id']}", axis=1
        )
        df["prompt"] = df.problem
        df = df.apply(lambda x: preprocess_uts_taco(x), axis=1)
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/taco_{split}.jsonl", orient="records", lines=True
        )
