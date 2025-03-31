set -x

# config files
critic_config_file="ctrl/gen/config/critic/default.yaml"
generator_config_file="ctrl/gen/config/generator/qwen25_coder_32b.yaml"
generator_greedy_config_file="ctrl/gen/config/generator/qwen25_coder_32b_greedy.yaml"

# model info
generator_model="Qwen/Qwen2.5-Coder-32B-Instruct"
critic_model="Critic"

# dataset arguments
dataset="scripts/data/code_contests/test.jsonl"

output_dir="res/code_contests"

mkdir -p $output_dir

export no_proxy=localhost,
export MAX_REQUESTS=256

# Define ckpts dictionary
declare -A ckpts=(
    ["critic"]="your_critic_ckpt"
)

# Base experiments
jobname="1_5"
if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
    ray stop &>/dev/null
    ray start --head &>/dev/null
    RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null

    echo "Generating ${output_dir}/${jobname}.jsonl"
    python3 scripts/run_gen.py \
        --output_file ${output_dir}/${jobname}.jsonl \
        --widths 1 5 \
        --dataset $dataset \
        --generator_config_file $generator_config_file \
        --critic_config_file $critic_config_file \
        --max_requests $MAX_REQUESTS
fi

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5.jsonl \
            --widths 1 5 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1_1_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1_1_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1_1_1_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 1 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1_1_1_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1_1_1_1_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 1 1 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1_1_1_1_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1_1_1_1_1_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 1 1 1 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done

# For each checkpoint in the dictionary
for model_name in "${!ckpts[@]}"; do
    jobname="1_5_1_1_1_1_1_1_1_1-critic-${model_name}"
    if ! hdfs dfs -test -e ${output_dir}/${jobname}.jsonl; then
        ray stop &>/dev/null
        ray start --head &>/dev/null
        if [[ "${ckpts[$model_name]}" = "hdfs"* ]];then
            if ! [ -d $model_name ]; then
                hdfs dfs get ${ckpts[$model_name]} $model_name
            fi
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            RAY_NUM_REPLICAS=1 RAY_JOB_NAME="critic" RAY_ROUTE_PREFIX="/critic" rayinfer vllm $model_name --served-model-name $critic_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=$critic_config_file
        else
            RAY_NUM_REPLICAS=2 RAY_JOB_NAME="generator" RAY_ROUTE_PREFIX="/generator" rayinfer vllm $generator_model --served-model-name $generator_model --enable-prefix-caching --tensor-parallel-size 4 &>/dev/null
            used_critic_config_file=${ckpts[$model_name]}
        fi

        echo "Generating ${output_dir}/${jobname}.jsonl"
        python3 scripts/run_gen.py \
            --output_file ${output_dir}/${jobname}.jsonl \
            --resume_file ${output_dir}/1_5_1_1_1_1_1_1_1-critic-${model_name}.jsonl \
            --widths 1 5 1 1 1 1 1 1 1 1 \
            --dataset $dataset \
            --generator_config_file $generator_greedy_config_file \
            --critic_config_file $used_critic_config_file \
            --max_requests $MAX_REQUESTS

        # Plot results
        python3 scripts/plot_sequential.py ${output_dir}/${jobname}.jsonl

        # Remove checkpoint to save space
        rm -rf $model_name
    fi
done
