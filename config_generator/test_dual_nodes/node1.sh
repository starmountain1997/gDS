#!/bin/bash
# Node 1 (Worker Node)

# ==================== Environment Variables ====================
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_MODELSCOPE="true"
export HCCL_BUFFSIZE="1024"
export SERVER_PORT="8080"
export OMP_PROC_BIND="true"
export OMP_NUM_THREADS="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_MLAPO="1"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="1"
export ASCEND_A3_EBA_ENABLE="1"

# ==================== Node Configuration ====================
# MASTER_IP needs to be replaced with Node 0's IP

# ==================== Startup Command ====================
vllm \
    serve \
    /mnt/weight/DeepSeek-V3.2-Exp-W8A8 \
    --served-model-name dsv3 \
    --headless \
    --data-parallel-size 4 \
    --data-parallel-rpc-port 13399 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 141.61.39.117 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --seed 1024 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 68000 \
    --max-num-batched-tokens 4096 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --speculative-config {"num_speculative_tokens": 3, "method":"deepseek_mtp"} \
    --compilation-config {"cudagraph_capture_sizes": [16, 32, 48, 64], "cudagraph_mode": "FULL_DECODE_ONLY"} \
    --additional-config {"layer_sharding": ["q_b_proj", "o_proj"]} \
    --tokenizer-mode deepseek_v32 \
    --reasoning-parser deepseek_v3
