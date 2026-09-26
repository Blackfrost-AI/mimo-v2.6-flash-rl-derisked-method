#!/usr/bin/env bash
set -euo pipefail

MODEL=/workspace/blackfrost/models/XiaomiMiMo/MiMo-V2.6-Flash-RL
VENV=/workspace/blackfrost/venvs/step5-patched-sglang-0.5.19
RUN=/workspace/blackfrost/research/mimo-v26-flash-thinking-on-iterative
INTERVENTION=${BLACKFROST_MIMO_MOE_INTERVENTION:?set the intervention manifest path}
PASS_NAME=${BLACKFROST_MIMO_PASS_NAME:?set the pass name}
MIMO_CUDA_VISIBLE_DEVICES=${BLACKFROST_MIMO_CUDA_VISIBLE_DEVICES:-4,5,6,7}
MIMO_PORT=${BLACKFROST_MIMO_PORT:-30001}

export HF_HOME=/workspace/.hf_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$MIMO_CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN/runtime-intervention-full${PYTHONPATH:+:$PYTHONPATH}"
export BLACKFROST_MIMO_MOE_INTERVENTION="$INTERVENTION"
export SGLANG_CACHE_DIR=/workspace/blackfrost/cache/sglang-mimo-v26-flash-clean
export TORCHINDUCTOR_CACHE_DIR=/workspace/blackfrost/cache/torchinductor-mimo-v26-flash-clean
export TRITON_CACHE_DIR=/workspace/blackfrost/cache/triton-mimo-v26-flash-clean

mkdir -p "$SGLANG_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$RUN/logs"

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "${BLACKFROST_MIMO_SERVED_MODEL_NAME:-Blackfrost/MiMo-V2.6-Flash-RL-$PASS_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --attention-backend fa4 \
  --mm-attention-backend fa4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.65 \
  --context-length 262144 \
  --max-running-requests 24 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --schedule-conservativeness 0.3 \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --host 127.0.0.1 \
  --port "$MIMO_PORT"
