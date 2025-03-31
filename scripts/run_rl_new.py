# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import asyncio
import json

import hydra
import pandas as pd
import ray
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import gsm8k, math

from ctrl.eval.score_utils import apply_verifiable_reward


def _default_compute_score(data_source, solution_str, ground_truth):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        return math.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError


class RewardManager:
    """The reward manager."""

    def __init__(
        self,
        files,
        tokenizer,
        num_examine,
        eos_penalty=False,
        compute_score=None,
        use_thinking=True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.eos_penalty = eos_penalty
        self.use_thinking = use_thinking

        if isinstance(files, str):
            self.dataset = pd.read_parquet(files)
        else:
            self.dataset = pd.concat([pd.read_parquet(f) for f in files])

        self.dataset["proxy_id"] = self.dataset["task_id"]
        self.id_to_infos = self.dataset.set_index("proxy_id").to_dict(orient="index")

    def check_format(self, response, eos_token=None):
        if self.use_thinking and (
            "</think>" not in response
            or response.count("</think>") > 1
            or response.startswith("</think>")
        ):
            return False, response

        if eos_token is not None:
            if eos_token not in response:
                return False, response

        response_without_think = (
            response.split("</think>")[1].strip() if self.use_thinking else response
        )
        if eos_token is not None:
            response_without_think = response_without_think.replace(eos_token, "")

        return True, response_without_think

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}
        solution_strs = []
        solution_without_think_strs = []
        infos = []
        datasets = []
        valid_response_lengths = []
        is_valids = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            solution_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False
            )

            # check format
            eos_token = self.tokenizer.eos_token if self.eos_penalty else None
            is_valid, solution_without_think_str = self.check_format(
                solution_str, eos_token=eos_token
            )

            # prepare data
            task_id = data_item.non_tensor_batch["task_id"]
            info = json.loads(self.id_to_infos[task_id]["info"])
            dataset = self.id_to_infos[task_id]["data_source"]  # NOTE: this is used in verl
            solution_strs.append(solution_str)
            infos.append(info)
            datasets.append(dataset)
            valid_response_lengths.append(valid_response_length)
            is_valids.append(is_valid)
            solution_without_think_strs.append(solution_without_think_str)

        # Define an async function to gather all scoring tasks
        async def compute_all_scores():
            tasks = []
            for sol, info, dataset in zip(solution_without_think_strs, infos, datasets):
                task = apply_verifiable_reward(sol, dataset, info)
                tasks.append(task)
            return await asyncio.gather(*tasks)

        # Use asyncio.run to execute the async function
        score_res = asyncio.run(compute_all_scores())
        scores = [res[0] for res in score_res]

        # Add format penalty
        scores = [
            -1 if not is_valid else score for score, is_valid in zip(scores, is_valids)
        ]

        for i, (score, dataset, valid_response_length) in enumerate(
            zip(scores, datasets, valid_response_lengths)
        ):
            reward_tensor[i, valid_response_length - 1] = score

            if dataset not in already_print_data_sources:
                already_print_data_sources[dataset] = 0

            if already_print_data_sources[dataset] < self.num_examine:
                already_print_data_sources[dataset] += 1
                print(solution_str)

        return reward_tensor


@hydra.main(config_path="rl_config_new", config_name="trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            }
        )

    ray.get(main_task.remote(config, compute_score))


@ray.remote
def main_task(config, compute_score=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # use custom chat template
    if config.data.chat_template is not None:
        # read jinja template
        with open(config.data.chat_template, "r") as f:
            tokenizer.chat_template = f.read()

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(
        files=config.data.train_files,
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=compute_score,
        eos_penalty=config.reward_model.eos_penalty,
        use_thinking=config.reward_model.use_thinking,
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        files=config.data.val_files,
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=compute_score,
        eos_penalty=config.reward_model.eos_penalty,
        use_thinking=config.reward_model.use_thinking,
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
