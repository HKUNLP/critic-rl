import argparse

import numpy as np
import pandas as pd

def get_updown(path, use_stop=False):
    df = pd.read_json(path, lines=True).dropna(subset=["success"])
    df_by_round = np.array_split(df, args.num_rounds + 1)
    last_success = df_by_round[-1].success.to_numpy()
    init_success = df_by_round[0].success.to_numpy()
    diff_last_round = last_success - init_success
    return init_success.mean(), last_success.mean(), (diff_last_round > 0).mean(), (diff_last_round < 0).mean()

def main(args):
    prev_pass1, pass1, up, down = get_updown(args.input, use_stop=args.use_stop)
    print(f"{args.input} {prev_pass1} {pass1} {up} {down}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--use_stop", action="store_true")
    parser.add_argument("--num_rounds", type=int, default=1)
    args = parser.parse_args()

    main(args)