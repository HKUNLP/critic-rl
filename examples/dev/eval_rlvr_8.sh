set -x

# config files
generator_config_file="ctrl/gen/config/generator/qwen25_32b.yaml"
generator_greedy_config_file="ctrl/gen/config/generator/qwen25_32b_greedy.yaml"

# model info
generator_model="Qwen/Qwen2.5-32B-Instruct"

# dataset arguments
dataset="scripts/data/rlvr/train.jsonl"

output_dir="res/rlvr"

mkdir -p $output_dir

export no_proxy=localhost,
export MAX_REQUESTS=256

# Base experiments
jobname="1_8"
if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
    ray stop &>/dev/null
    ray start --head &>/dev/null
    RAY_NUM_REPLICAS=4 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 2 &>/dev/null

    echo "Generating ${output_dir}/${jobname}.jsonl"
    python3 scripts/run_gen.py \
        --output_file ${output_dir}/${jobname}.jsonl \
        --widths 1 8 \
        --dataset $dataset \
        --generator_config_file $generator_config_file \
        --max_requests $MAX_REQUESTS
fi
