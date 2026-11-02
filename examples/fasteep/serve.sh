#!/usr/bin/env bash
# Start vLLM at the maximum EP size so FastEEP can shrink and grow it later.
#
#   MODEL=/models/DeepSeek-V2-Lite ./serve.sh
#   MODEL=/models/Qwen3-30B-A3B DP=8 PORT=8005 ./serve.sh
#
# All flags below are the validated FastEEP configuration: sleep mode, EPLB,
# ElasticEP routing, NIXL EP all2all, PyNCCL weight migration, Ray DP.
set -euo pipefail

MODEL="${MODEL:?set MODEL to a local model path or HF id}"
DP="${DP:-8}"                 # maximum DP/EP size: the ceiling for scaling
PORT="${PORT:-8005}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
REDUNDANT_EXPERTS="${REDUNDANT_EXPERTS:-64}"
RAY_PORT="${RAY_PORT:-9305}"

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_FLASH_EPSCALE_SUSPEND_MAX_DP="${VLLM_FLASH_EPSCALE_SUSPEND_MAX_DP:-1}"
export VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN="${VLLM_FLASH_EPSCALE_SKIP_NIXL_TEARDOWN:-0}"

if ! timeout 10s ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    ray start --head --port="${RAY_PORT}" --disable-usage-stats
fi
ray status --address="${RAY_ADDRESS}"

echo "serving ${MODEL} with DP/EP=${DP} on :${PORT} (max size; scale with demo.py)"
exec vllm serve "${MODEL}" \
    --trust-remote-code \
    --enable-sleep-mode \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-eplb \
    --enable-elastic-ep \
    --all2all-backend nixl_ep \
    --eplb-config.num_redundant_experts "${REDUNDANT_EXPERTS}" \
    --eplb-config.communicator pynccl \
    --eplb-config.use_async false \
    --data-parallel-backend ray \
    --distributed-executor-backend ray \
    --data-parallel-size "${DP}" \
    --data-parallel-size-local "${DP}" \
    --data-parallel-rpc-port 9876 \
    --data-parallel-start-rank 0 \
    --enforce-eager
