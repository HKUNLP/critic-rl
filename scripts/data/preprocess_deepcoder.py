import json
import os

import numpy as np
from datasets import load_dataset


def preprocess_uts(tests):
    tests = json.loads(tests)
    new_tests = []

    for t in tests:
        t['input'] = {'stdin': t['input']}
        t['output'] = {'stdout': t['output']}
        new_tests.append(t)
    return json.dumps(new_tests)


def preprocess_uts_taco(tests):
    tests = json.loads(tests)
    new_tests = []

    for i, o in zip(tests['inputs'], tests['outputs']):
        t = {}
        t['input'] = {'stdin': i}
        t['output'] = {'stdout': o}
        new_tests.append(t)
    return json.dumps(new_tests)

if __name__ == "__main__":
    os.makedirs("scripts/data/deepcoder", exist_ok=True)
    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "codeforces")
    
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(lambda x: f"deepscaler-codeforces/{split}/{x['task_id']}", axis=1)
        df["dataset"] = "code_contests"
        df["prompt"] = df.problem
        df["info"] = df.apply(
            lambda x: json.dumps({"test": preprocess_uts(x["tests"])}), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/codeforces_{split}.jsonl", orient="records", lines=True
        )

    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5")
    
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(lambda x: f"deepscaler-lcbv5/{split}/{x['task_id']}", axis=1)
        df["dataset"] = "code_contests"
        df["prompt"] = df.problem
        df["info"] = df.apply(
            lambda x: json.dumps({"test": preprocess_uts(x["tests"])}), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/lcbv5_{split}.jsonl", orient="records", lines=True
        )

    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect")
    
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(lambda x: f"deepscaler-primeintellect/{split}/{x['task_id']}", axis=1)
        df["dataset"] = "code_contests"
        df["prompt"] = df.problem
        df["info"] = df.apply(
            lambda x: json.dumps({"test": preprocess_uts(x["tests"])}), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/primeintellect_{split}.jsonl", orient="records", lines=True
        )

    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "taco")
    
    for split in dataset.keys():
        df = dataset[split].to_pandas()
        df["task_id"] = np.arange(len(df))
        df["task_id"] = df.apply(lambda x: f"deepscaler-taco/{split}/{x['task_id']}", axis=1)
        df["dataset"] = "code_contests"
        df["prompt"] = df.problem
        df["info"] = df.apply(
            lambda x: json.dumps({"test": preprocess_uts_taco(x["tests"])}), axis=1
        )
        df[["task_id", "prompt", "dataset", "info"]].to_json(
            f"scripts/data/deepcoder/taco_{split}.jsonl", orient="records", lines=True
        )