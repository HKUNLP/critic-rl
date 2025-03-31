# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import json
import random

import pandas as pd
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio

from ctrl.eval.score_utils import apply_verifiable_reward
from ctrl.gen.api import (
    async_chat_api_response,
    build_singleturn_messages,
)
from ctrl.gen.prompt import get_prompter
from ctrl.gen.tree import NodeType, Node, Tree


async def main(args):
    # load dataset
    raw_df = pd.read_json(
        args.dataset,
        lines=True,
    )
    raw_df = raw_df.head(args.num_samples) if args.num_samples is not None else raw_df
    ds = {
        idx: {"task_id": idx, **data}
        for idx, data in raw_df.set_index("task_id").to_dict(orient="index").items()
    }

    # construct tree
    if args.resume_file is not None:
        resume_df = pd.read_json(args.resume_file, lines=True)
        tree = Tree.from_dataframe(resume_df)
    else:
        tree = Tree()
        root_nodes = [
            Node(
                task_id=task_id,
                hash=Tree.generate_random_hash(),
                node_type=NodeType.ROOT,
            )
            for task_id in ds.keys()
        ]
        tree.add_nodes(root_nodes, depth=0)

    # load generator
    generator_conf = OmegaConf.load(args.generator_config_file)

    # load critic (optional)
    if args.critic_config_file is not None:
        critic_conf = OmegaConf.load(args.critic_config_file)

    async def async_response_with_semaphore(messages, conf, semaphore=None):
        async with semaphore if semaphore is not None else asyncio.Semaphore(1):
            response = await async_chat_api_response(messages, **conf)
        return response

    # MAIN LOOP: update responses for multiple iterations
    target_widths = args.widths
    nodes_to_expand = tree.expand(target_widths)
    while nodes_to_expand:
        for depth, nodes in nodes_to_expand.items():  # NOTE: currently only one level
            # prepare data
            task_ids = [s.task_id for s in nodes]
            problems = [ds[task_id]["prompt"] for task_id in task_ids]
            datasets = [ds[task_id]["dataset"] for task_id in task_ids]
            solutions = [s.solution for s in nodes]
            node_type = nodes[0].node_type  # same for all

            # construct messages for new solutions
            critiques = None
            if node_type == NodeType.ROOT:
                gen_prompts = [
                    get_prompter(dataset).get_gen_prompt(p)
                    for p, dataset in zip(problems, datasets)
                ]
                new_messages = [
                    build_singleturn_messages(p, args.generator_system_prompt)
                    for p in gen_prompts
                ]

                # inference
                # use semaphore to limit the number of requests
                semaphore = asyncio.Semaphore(args.max_requests)

                async def direct_generate(node, messages):
                    response = await async_response_with_semaphore(
                        messages, generator_conf, semaphore
                    )

                    # get execution results
                    task_id = node.task_id
                    sample = ds[task_id]
                    dataset = sample["dataset"]
                    info = json.loads(sample["info"]) if isinstance(sample["info"], str) else sample["info"]
                    success, metadata = await apply_verifiable_reward(
                        response, dataset=dataset, info=info
                    )

                    return Node(
                        task_id=task_id,
                        solution=response,
                        critique=None,
                        node_type=Tree.next_node_type(node_type),
                        success=success,
                        metadata=metadata,
                        prev_hash=node.hash,
                        hash=Tree.generate_random_hash(),
                    )

                new_nodes = await tqdm_asyncio.gather(
                    *[
                        direct_generate(node, messages)
                        for node, messages in zip(nodes, new_messages)
                    ],
                    desc=f"Generating new solutions for depth {depth}",
                    leave=False,
                )
            else:  # critique-then-correct
                # use semaphore to limit the number of requests
                semaphore = asyncio.Semaphore(args.max_requests)

                async def critique_then_correct(node, problem, solution, dataset):
                    prompter = get_prompter(dataset)
                    async with semaphore:
                        if args.critic_mode == "critique":
                            critic_prompt = prompter.get_critique_prompt(
                                problem, solution
                            )
                        else:
                            eval_res = node.metadata["result"]
                            critic_prompt = prompter.get_critique_with_exec_prompt(
                                problem, solution, eval_res
                            )
                        messages = build_singleturn_messages(
                            critic_prompt, args.critic_system_prompt
                        )
                        critique = await async_response_with_semaphore(
                            messages, critic_conf
                        )

                        if isinstance(critique, list):
                            judges = [
                                "Overall judgment: Correct" in c for c in critique
                            ]
                            major_judge = sum(judges) > (len(judges) / 2)

                            # randomly select one
                            critique = random.choice(
                                [
                                    c
                                    for i, c in enumerate(critique)
                                    if judges[i] == major_judge
                                ]
                            )

                        if args.critic_reasoning:
                            if "</think>" not in critique:
                                critique = ""
                            else:
                                critique = critique.split("</think>")[1].strip()

                        # construct messages for new solutions
                        gen_prompt = prompter.get_gen_prompt(problem)
                        revise_prompt = prompter.get_revised_prompt_mt(critique)
                        new_messages = build_singleturn_messages(
                            gen_prompt, args.generator_system_prompt
                        ) + [
                            {
                                "role": "assistant",
                                "content": solution,
                            },
                            {
                                "role": "user",
                                "content": revise_prompt,
                            },
                        ]
                        response = await async_response_with_semaphore(
                            new_messages, generator_conf
                        )

                    # get execution results
                    task_id = node.task_id
                    sample = ds[task_id]
                    dataset = sample["dataset"]
                    info = json.loads(sample["info"]) if isinstance(sample["info"], str) else sample["info"]
                    success, metadata = await apply_verifiable_reward(
                        response, dataset=dataset, info=info
                    )

                    return Node(
                        task_id=task_id,
                        solution=response,
                        critique=critique,
                        node_type=Tree.next_node_type(node_type),
                        success=success,
                        metadata=metadata,
                        prev_hash=node.hash,
                        hash=Tree.generate_random_hash(),
                    )

                # inference
                new_nodes = await tqdm_asyncio.gather(
                    *[
                        critique_then_correct(node, problem, solution, dataset)
                        for node, problem, solution, dataset in zip(
                            nodes, problems, solutions, datasets
                        )
                    ],
                    desc=f"Generating new solutions for depth {depth} by critique-then-correct",
                    leave=False,
                )

            # update
            tree.add_nodes(new_nodes, depth)

        nodes_to_expand = tree.expand(target_widths)

        # save results at each iteration
        df = tree.to_dataframe()
        df.to_json(args.output_file, lines=True, orient="records", force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_file", type=str, help="output file name")
    parser.add_argument(
        "--resume_file", type=str, default=None, help="resume file name"
    )
    parser.add_argument("--max_requests", type=int, default=256)

    # data parameters
    parser.add_argument("--num_samples", type=int, default=None)

    # generation parameters
    parser.add_argument("--generator_config_file", type=str, required=True)
    parser.add_argument("--generator_system_prompt", type=str, default=None)

    # critic parameters (optional)
    parser.add_argument(
        "--critic_mode",
        type=str,
        choices=["critique", "critique-with-exec"],
        default="critique",
    )
    parser.add_argument("--critic_config_file", type=str, default=None)
    parser.add_argument("--critic_system_prompt", type=str, default=None)
    parser.add_argument("--critic_reasoning", action="store_true")

    # tree arguments
    parser.add_argument("--widths", nargs="+", type=int)

    # parse arguments
    args = parser.parse_args()

    # let's go!
    asyncio.run(main(args))
