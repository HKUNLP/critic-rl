import argparse
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ctrl.gen.tree import NodeType, Node, Tree

sns.set_style("whitegrid")


def get_all_node_groups(tree: Tree) -> Dict[int, Dict[int, List[Node]]]:
    """task id -> (group configuration -> groups)"""
    # construct mapping
    children_map = defaultdict(list)
    for node in sum(tree.nodes[1:], []):
        children_map[node.prev_hash].append(node)

    # construct groups
    all_groups = dict()
    for root_node in tree.nodes[0]:
        task_groups = defaultdict(list)
        gen_nodes = children_map[root_node.hash]

        for gen_node in gen_nodes:
            task_groups[0].append([gen_node])
            curr_node = gen_node
            final_node = gen_node
            stopped = False
            depth = 1

            while curr_node.hash in children_map:
                next_node = children_map[curr_node.hash][0]
                # if stopped or ("Overall judgment: Correct" in next_node.critique):
                # if stopped or (not next_node.sanitized_solution.startswith("```")):
                # if stopped or final_node.success:  # NOTE: just for analysis
                if False:
                    task_groups[depth].append([final_node])
                    stopped = True
                else:
                    task_groups[depth].append([next_node])
                    final_node = next_node

                curr_node = next_node
                depth += 1

        all_groups[root_node.task_id] = dict(task_groups)

    return all_groups


def cal_success_by_group(groups: List[List[Node]]) -> Tuple[List[int], List[bool]]:
    """Return steps + success rate for each group"""
    steps = []
    is_success = []

    for group in groups:
        steps.append(len(group))
        is_success.append(any([node.success for node in group]))

    cum_steps = np.cumsum(steps).tolist()
    cum_success = (np.cumsum(is_success) > 0).tolist()

    return cum_steps, cum_success


def main(args):
    # load tree
    df = pd.read_json(args.input, lines=True)
    df.metadata = {}  # we don't need metadata for this analysis
    tree = Tree.from_dataframe(df)

    # shuffle to avoid randomness
    all_res = defaultdict(list)
    repeated_config_to_steps = defaultdict(list)
    repeated_config_to_success = defaultdict(list)
    repeated_config_to_up = defaultdict(list)
    repeated_config_to_down = defaultdict(list)
    for i in range(args.num_shuffles):
        tree_ = deepcopy(tree)
        tree_.shuffle(seed=i)

        # get all groups
        all_groups = get_all_node_groups(tree_)

        # cal success rate
        config_to_steps = defaultdict(list)
        config_to_success = defaultdict(list)
        config_to_up = defaultdict(list)
        config_to_down = defaultdict(list)
        for task_id, task_groups in all_groups.items():
            for config, groups in task_groups.items():
                cum_steps, cum_success = cal_success_by_group(groups)
                config_to_steps[config].append(cum_steps)
                config_to_success[config].append(cum_success)

                _, init_cum_success = cal_success_by_group(task_groups[0])
                config_to_up[config].append([(success - init_success) > 0 for success, init_success in zip(cum_success, init_cum_success)])
                config_to_down[config].append([(success - init_success) < 0 for success, init_success in zip(cum_success, init_cum_success)])

                num_groups = len(groups)
                all_res["task_id"].extend([task_id] * num_groups)
                all_res["config"].extend([config] * num_groups)
                all_res["steps"].extend(cum_steps)
                all_res["success"].extend(cum_success)
                all_res["seed"].extend([i] * len(cum_steps))

                # # add updown analysis
                # all_res["up_success"].extend([(success - init_success) > 0 for success, init_success in zip(cum_success, init_cum_success)])
                # all_res["down_success"].extend([(success - init_success) < 0 for success, init_success in zip(cum_success, init_cum_success)])

        # average over tasks
        config_to_steps = {k: np.mean(v, axis=0) for k, v in config_to_steps.items()}
        config_to_success = {
            k: np.mean(v, axis=0) for k, v in config_to_success.items()
        }
        config_to_up = {k: np.mean(v, axis=0) for k, v in config_to_up.items()}
        config_to_down = {k: np.mean(v, axis=0) for k, v in config_to_down.items()}
        for k, v in config_to_steps.items():
            repeated_config_to_steps[k].append(v)
        for k, v in config_to_success.items():
            repeated_config_to_success[k].append(v)
        for k, v in config_to_up.items():
            repeated_config_to_up[k].append(v)
        for k, v in config_to_down.items():
            repeated_config_to_down[k].append(v)

    # average over repeats
    config_to_steps = {
        k: np.mean(v, axis=0) for k, v in repeated_config_to_steps.items()
    }
    config_to_success = {
        k: np.mean(v, axis=0) for k, v in repeated_config_to_success.items()
    }
    config_to_up = {k: np.mean(v, axis=0) for k, v in repeated_config_to_up.items()}
    config_to_down = {k: np.mean(v, axis=0) for k, v in repeated_config_to_down.items()}

    # plot
    palette = sns.color_palette("vlag_r")
    config_rename = {
        0: "Repeated Sampling (Baseline)",
        1: "Critique-Correction",
        2: "Critique-Correction x2",
    }
    color_map = {
        "Repeated Sampling (Baseline)": palette[-1],
        "Critique-Correction": palette[0],
        "Critique-Correction x2": palette[1],
    }
    data = pd.DataFrame(all_res)
    data["config"] = data["config"].map(config_rename)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(
        data=data, x="steps", y="success", hue="config", ax=ax, palette=color_map
    )
    # for config, cum_steps in config_to_steps.items():
    #     plt.plot(cum_steps, config_to_success[config], label=f"{config} Layer")
    #     ax.set_xscale('log')
    ax.set_xscale("log", base=2)
    plt.xlabel("Sample Budget (K)")
    plt.ylabel("Pass@K")
    plt.legend()

    # organize data for saving
    data = defaultdict(list)
    for config, cum_steps in config_to_steps.items():
        data["steps"].extend(cum_steps.tolist())
        data["success"].extend(config_to_success[config].tolist())
        data["up"].extend(config_to_up[config].tolist())
        data["down"].extend(config_to_down[config].tolist())
        data["config"].extend([config] * len(cum_steps))

    df = pd.DataFrame(data)

    # Interpolate separately for each configuration
    interpolated_data = []

    for config in df["config"].unique():
        config_data = df[df["config"] == config].copy()

        # Create a DataFrame with all steps for this configuration
        full_steps_df = pd.DataFrame(
            {
                "steps": np.arange(
                    min(config_data["steps"]), max(config_data["steps"]) + 1
                )
            }
        )

        # Merge with existing data
        config_data = full_steps_df.merge(config_data, on="steps", how="left")

        # Interpolate success values
        config_data["success"] = config_data["success"].interpolate(method="linear")
        config_data["config"] = config

        interpolated_data.append(config_data)

    df = pd.concat(interpolated_data, ignore_index=True)
    print(df.to_markdown()) 

    # save
    if args.output is not None:
        df.to_json(args.output.replace(".pdf", ".jsonl"), orient="records", lines=True)
        pd.DataFrame(all_res).to_json(
            args.output.replace(".pdf", "_all.jsonl"), orient="records", lines=True
        )
        plt.tight_layout()
        plt.savefig(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--num_shuffles", type=int, default=1, help="Number of shuffles"
    )
    parser.add_argument("--difficulty_field", type=str, default=None)

    args = parser.parse_args()

    main(args)