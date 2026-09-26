#!/usr/bin/env bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

run_dir=/workspace/blackfrost/research/mimo-v26-flash-thinking-on-iterative
export BLACKFROST_MIMO_MOE_INTERVENTION="${run_dir}/interventions/pass12-drowzeys-experts-l28-33-iteration2-lambda3.5.json"
export BLACKFROST_MIMO_PASS_NAME=pass12-drowzeys-experts-l28-33-iteration2-lambda3.5
export BLACKFROST_MIMO_CUDA_VISIBLE_DEVICES=0,1,2,3
export BLACKFROST_MIMO_PORT=30002
export BLACKFROST_MIMO_SERVED_MODEL_NAME="MiMo-v2.6-Flash-RL-Derisked"

exec bash "${run_dir}/scripts/launch_intervention.sh"
