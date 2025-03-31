set -x

critic_config_file="ctrl/gen/config/generator/qwen25_coder_32b.yaml"  # using the generator itself
generator_config_file="ctrl/gen/config/generator/qwen25_coder_32b.yaml"
generator_config_greedy_file="ctrl/gen/config/generator/qwen25_coder_32b_greedy.yaml"
generator_model=Qwen/Qwen2.5-Coder-32B-Instruct

output_dir="scripts/data/gen_data"
jobname="qwen25-32b-train-alluts"

mkdir -p $output_dir

export no_proxy=localhost,
export n_samples=4  # specifying how many initial solutions to sample per question

ray stop; ray start --head
RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4

declare -A prompter_map
prompter_map[io]="code_contests"
prompter_map[assert]="mbppplus"
for split in "io" "assert"; do
    echo "Running $split"

    python3 scripts/run_gen.py \
        --output_file ${output_dir}/${jobname}-${split}-zeroshot.jsonl \
        --problem_field problem \
        --test_field all_uts \
        --width 1 $n_samples \
        --dataset scripts/data/taco/train_$split.jsonl \
        --generator_config_file $generator_config_file \
        --generator_system_prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
        --prompter_type ${prompter_map[$split]} \
        --num_samples 1000 \
        --max_requests 128

    python3 scripts/run_gen.py \
        --output_file ${output_dir}/${jobname}-${split}-critique-with-exec.jsonl \
        --resume_file ${output_dir}/${jobname}-${split}-zeroshot.jsonl \
        --problem_field problem \
        --test_field all_uts \
        --critic_mode critique-with-exec \
        --width 1 $n_samples 1 \
        --dataset scripts/data/taco/train_$split.jsonl \
        --generator_config_file $generator_config_greedy_file \
        --generator_system_prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
        --critic_config_file $critic_config_file \
        --critic_system_prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
        --prompter_type ${prompter_map[$split]} \
        --num_samples 1000 \
        --max_requests 128

    # convert data for sft training
    python3 scripts/data/cleanup_sft.py \
        --dataset scripts/data/taco/train_$split.jsonl \
        --output_file sft_32b_${split}.jsonl \
        --resume_file ${output_dir}/${jobname}-${split}-critique-with-exec.jsonl \
        --prompter_type ${prompter_map[$split]} \
        --max_prompt_len 2048 \
        --keep_failed \
        --remove_duplicates

    # convert data for rl training
    # NOTE: there is overlapping data for sft and rl
    # for better performance, you can explicitly exclude sft data from rl
    python3 scripts/data/cleanup_rl.py \
        ${output_dir}/${jobname}-${split}-zeroshot.jsonl \
        ${jobname}-${split}.parquet \
        --dataset_path scripts/data/taco/train_${split}.jsonl \
        --keep_chat
done

cat sft_32b_assert.jsonl sft_32b_io.jsonl > sft_32b.jsonl

split="assert"
python3 scripts/data/cleanup_rl.py \
    ${output_dir}/${jobname}-${split}-zeroshot.jsonl \
    ${jobname}-${split}.parquet \
    --dataset_path scripts/data/taco/train_${split}.jsonl \
    --split_name ${split} \
    --keep_chat \
    --prompter_type $split