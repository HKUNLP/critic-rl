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
import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import List

import evaluate

from ctrl.eval.deepscaler_utils import (
    RewardMathFn,
    RewardConfig,
    RewardInput,
    RewardType,
)
from ctrl.eval.ifeval_utils import IF_FUNCTIONS_MAP
from ctrl.eval.ifeval_ori_utils import evaluation_lib as ifeval_lib
from ctrl.eval.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    last_boxed_only_string_v2,
    normalize_final_answer,
    remove_boxed,
)
from ctrl.eval.orz_utils import is_equal, solution2answer
from ctrl.eval.sandbox_utils import get_submit_fn, submit_to_sandbox


rouge = evaluate.load("rouge")

def verify_gsm8k_sample(model_output, ground_truth_answer):
    assert ground_truth_answer is not None, "ground truth answer is None"
    # gsm is easy: extract numbers, and then just compare last number with answer.
    # matches how we do eval.
    predictions = None
    # replace numbers like `x,xxx` with `xxxx`
    response = re.sub(r"(\d),(\d)", r"\1\2", model_output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    if numbers:
        predictions = numbers[-1]
    else:
        predictions = response
    return str(predictions).lower() == str(ground_truth_answer).lower()


def verify_math_sample(model_output, ground_truth_answer):
    assert ground_truth_answer is not None, "ground truth answer is None"
    raw_answer = model_output
    # for math, more complex. We will try a few different ways to extract the answer.
    # this roughly follows 'flex em' in oe-eval-internal
    all_answers = []
    # First, try find answer in \boxed{}.
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # If nothing still, try to find the last latex-formatted answer
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
        if len(dollars) > 1:
            # Add the answer between the second to last and last dollar sign
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False
    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched


def verify_strict_math_sample(model_output, ground_truth_answer):
    raw_answer = model_output
    # just trying minerva format.
    all_answers = []
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False
    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched


def verify_ifeval_sample(model_output, constraint):
    assert constraint is not None, "constraint is None"
    # TODO: just pass in final answer. this should be fine for other evals too.
    answer = model_output.split("<|assistant|>\n")[-1].strip()
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    if "func_name" not in constraint:
        print("WARNING: constraint missing func_name")
        print(constraint)
        return False
    # first, parse out the constraint string.
    func_name = constraint.pop("func_name")
    # get the function
    func = IF_FUNCTIONS_MAP[func_name]
    # now, run the function
    # pop out any none args
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    # sometimes we have extra args, sometimes not.
    if len(constraint) == 0:
        return func(model_output)
    return func(answer, **non_none_args)


def verify_gpqa_sample(model_output, ground_truth_answer):
    pred = last_boxed_only_string_v2(model_output[-100:])
    if pred is None:
        return False
    pred = remove_boxed(pred)
    return pred.upper() == ground_truth_answer


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    From https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def verify_flan_sample(model_output, ground_truth_answer):
    # Flan! we will just use... exact match with some basic cleaning, after extracting the answer.
    answer_string = model_output.split("The answer is: ")[-1].strip()
    return normalize_answer(answer_string) == normalize_answer(ground_truth_answer)


def soft_format_reward_func(
    responses: list[str], reward_scale: float = 1.0
) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [reward_scale if match else 0.0 for match in matches]


async def verify_code_contests_sample(model_output, info):
    req = get_submit_fn("code_contests")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.accepted, res.dict()


async def verify_livecodebench_sample(model_output, info):
    req = get_submit_fn("livecodebench")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.accepted, res.dict()


async def verify_mbppplus_sample(model_output, info):
    req = get_submit_fn("mbppplus")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.status == "Success", res.dict()


async def apply_verifiable_reward(response: str, dataset: str, info: dict):
    if dataset.lower() == "gsm8k":
        reward = verify_gsm8k_sample(response, info["ground_truth"])
        metadata = {"dataset": "gsm8k"}
    elif dataset.lower() == "math":
        reward = verify_math_sample(response, info["ground_truth"])
        metadata = {"dataset": "math"}
    elif dataset.lower() == "aime24":
        reward = verify_math_sample(response, info["ground_truth"])
        metadata = {"dataset": "aime24"}
    elif dataset.lower() == "aime25":
        reward = verify_math_sample(response, info["ground_truth"])
        metadata = {"dataset": "aime25"}
    elif dataset.lower() == "gpqa":
        reward = verify_gpqa_sample(response, info["ground_truth"])
        metadata = {"dataset": "gpqa"}
    elif dataset.lower() == "ifeval":
        reward = verify_ifeval_sample(response, info["ground_truth"])
        metadata = {"dataset": "ifeval"}
    elif dataset.lower() == "code_contests":
        reward, metadata = await verify_code_contests_sample(response, info)
        metadata["dataset"] = "code_contests"
    elif dataset.lower() == "livecodebench":
        reward, metadata = await verify_livecodebench_sample(response, info)
        metadata["dataset"] = "livecodebench"
    elif dataset.lower() == "mbppplus":
        reward, metadata = await verify_mbppplus_sample(response, info)
        metadata["dataset"] = "mbppplus"
    elif dataset.lower() == "deepscaler":
        reward_fn = RewardMathFn(RewardConfig)
        inp = RewardInput(
            problem="",
            problem_type=RewardType.MATH,
            model_response=response,
            ground_truth={"answer": info["ground_truth"]},
        )
        reward = reward_fn(inp).is_correct
        metadata = {"dataset": "deepscaler"}
    elif dataset.lower() == "orz":
        pattern = re.compile(r"(\\boxed{.*})", re.DOTALL)
        matches = re.findall(pattern, response)
        final_answer = matches[-1] if matches else ""
        reward = await is_equal(
            solution2answer(info["ground_truth"]),
            solution2answer(final_answer),
            ThreadPoolExecutor(max_workers=64),
        )
        reward = float(reward)
        metadata = {"dataset": "orz"}
    elif dataset.lower() == "tldr":
        response = response.split("TL;DR:")[-1].strip()
        rouge_res = rouge.compute(
            predictions=[response], references=[info["ground_truth"]]
        )
        reward = rouge_res["rouge1"]
        metadata = {"dataset": "tldr"} | rouge_res
    elif dataset.lower() == "ifeval_ori":
        inp = ifeval_lib.InputExample(**info)
        prompt = info["prompt"]
        prompt_to_response = {prompt: response}
        metadata = {"dataset": "ifeval_ori"}

        # get instruction following results
        for func, output_file_name in [
            (ifeval_lib.test_instruction_following_strict, "eval_results_strict"),
            (ifeval_lib.test_instruction_following_loose, "eval_results_loose"),
        ]:
            output = func(inp, prompt_to_response)
            follow_all_instructions = output.follow_all_instructions
            metadata[output_file_name] = follow_all_instructions
        reward = metadata["eval_results_strict"]
    
    return reward, metadata
