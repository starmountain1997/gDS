#!/bin/bash
# vLLM Server Startup Script

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND="false"
export OMP_NUM_THREADS="1"
export HCCL_BUFFSIZE="1024"
export VLLM_ASCEND_ENABLE_MLAPO="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="1"

# ==================== Server Configuration ====================
# Model: /mnt/weight/DeepSeek-V3.2-Exp-W8A8
# Served Model Name: dsv3
# Tensor Parallel: 8
# Data Parallel: 2
# Port: 8087

# ==================== Startup Command ====================
vllm serve /mnt/weight/DeepSeek-V3.2-Exp-W8A8 \
    --served-model-name dsv3 \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --port 8087 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --trust-remote-code \
    --quantization ascend \
    --gpu-memory-utilization 0.98 \
    --compilation-config '{"cudagraph_capture_sizes":[8, 16], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
    --reasoning-parser deepseek_v3 \
    --tokenizer_mode deepseek_v32
