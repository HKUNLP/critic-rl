import argparse
import asyncio
import json
import re

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from ctrl.eval.score_utils import apply_verifiable_reward
from ctrl.gen.api import async_chat_api_response
from ctrl.gen.prompt import get_prompter


class MessageTracker:
    critique_prompt = "Analyze your previous response, judge whether it needs revision, and then provide a finalized response using the same format."

    def __init__(
        self, api_key, base_url, prompt, model, dataset, info, output_judgment=True
    ):
        self.messages = []
        self.raw_trajectories = []
        self.stopped = False

        # format prompt
        self.next_prompt = get_prompter(dataset).get_gen_prompt(prompt)
        self.output_judgment = output_judgment

        # define generate_fn
        async def generate_fn(messages, **kwargs):
            return await async_chat_api_response(
                messages, model, api_key, base_url, **kwargs
            )

        self.generate_fn = generate_fn

        # define verify_fn
        async def verify_fn(response):
            reward = await apply_verifiable_reward(
                response=response, dataset=dataset, info=info
            )
            return reward

        self.verify_fn = verify_fn

    @staticmethod
    def format_gen(prompt):
        return f"""{prompt}"""

    def format_critique(self, prompt):
        if not self.output_judgment:
            f"""{prompt}"""

        return f"""{prompt}"""

    @staticmethod
    def extract_gen_response(response):
        pattern = r"(.+?)\n</think>\n\n(.+)$"
        match = re.search(pattern, response.strip(), re.DOTALL | re.VERBOSE)

        if not match:
            return (False, {})

        thoughts, response = match.groups()
        return (True, {"thoughts": thoughts.strip(), "response": response.strip()})

    @staticmethod
    def extract_critique_response(response):
        pattern = (
            r"(.+?)\n</think>\n\n(.+)$"
        )
        match = re.search(pattern, response.strip(), re.DOTALL | re.VERBOSE)

        if not match:
            return (False, {})

        thoughts, response = match.groups()
        return (
            True,
            {
                "thoughts": thoughts.strip(),
                "response": response.strip(),
            },
        )

    async def generate_and_verify_next_with_semaphore(self, semaphore=None, **kwargs):
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)

        async with semaphore:
            return await self.generate_and_verify_next(**kwargs)

    async def generate_and_verify_next(self, **kwargs):
        # we only generate at the first turn
        extract_fn = (
            self.extract_gen_response
            if len(self.messages) == 0
            else self.extract_critique_response
        )
        format_fn = self.format_gen if len(self.messages) == 0 else self.format_critique

        # generate
        self.messages.append(
            {"role": "user", "content": format_fn(prompt=self.next_prompt)}
        )
        responses = await self.generate_fn(self.messages, **kwargs)

        # extract response
        formatted, output = zip(*[extract_fn(r) for r in responses])

        if not any(formatted):
            print(f"No valid found in response: {responses[0]}")
            self.stopped = True
            return False, 0.0

        verify_res = await asyncio.gather(
            *[(self.verify_fn(o["response"])) for o, f in zip(output, formatted) if f],
        )
        rewards, metadatas = zip(*verify_res)
        self.raw_trajectories.append(
            dict(
                prompt=self.next_prompt,
                responses=responses,
                formatted=formatted,
                output=output,
                rewards=rewards,
            )
        )

        # check if some of the responses are correct
        if any(rewards):
            self.stopped = True
            return False, sum(rewards) / sum(formatted)

        # randomly pick one response to expand
        self.messages[-1]["content"] = self.next_prompt
        self.messages.append(
            {
                "role": "assistant",
                "content": [o["response"] for o, f in zip(output, formatted) if f][0],
            }
        )

        # next prompt is the critique prompt
        self.next_prompt = self.critique_prompt

        return True, 0.0


async def main(args):
    raw_df = pd.read_json(
        args.input,
        lines=True,
    )
    if args.num_samples is not None:
        raw_df = raw_df.sample(n=args.num_samples)

    all_trackers = []
    for row in raw_df.itertuples():
        prompt, dataset, info = row.prompt, row.dataset, json.loads(row.info)
        all_trackers.append(
            MessageTracker(
                "empty",
                "http://localhost:8000/generator/v1",
                prompt,
                args.model,
                dataset,
                info,
                output_judgment=not args.disable_judgment,
            )
        )

    semaphore = asyncio.Semaphore(256)
    for _ in range(args.max_rounds):
        await tqdm_asyncio.gather(
            *[
                tracker.generate_and_verify_next_with_semaphore(
                    n=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    semaphore=semaphore,
                    max_tokens=args.max_tokens,
                )
                for tracker in all_trackers
                if not tracker.stopped
            ]
        )

    if args.output is not None:
        all_res = []
        for tracker in all_trackers:
            all_res.append(tracker.raw_trajectories)
        with open(args.output, "w") as f:
            for res in all_res:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    # generation args
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--disable_judgment", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args))
