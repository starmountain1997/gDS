#!/bin/bash
# Node 0 (Master Node)

# ==================== Environment Variables ====================
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_MODELSCOPE="true"
export HCCL_BUFFSIZE="1024"
export SERVER_PORT="8080"
export OMP_PROC_BIND="true"
export OMP_NUM_THREADS="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="1"
export ASCEND_A3_EBA_ENABLE="1"

# ==================== Node Configuration ====================
# LOCAL_IP needs to be replaced with actual machine IP

# ==================== Startup Command ====================
vllm \
    serve \
    vllm-ascend/DeepSeek-V3.2-W8A8 \
    '--served-model-name dsv3' \
    --host 0.0.0.0 \
    --port '$SERVER_PORT' \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-address '$LOCAL_IP' \
    --data-parallel-rpc-port 13399 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --seed 1024 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
    --compilation-config '{"cudagraph_capture_sizes": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
    --tokenizer-mode deepseek_v32 \
    --reasoning-parser deepseek_v3
