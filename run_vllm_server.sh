export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1

vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8080 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --dtype bfloat16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --max-num-seqs 1 \
  --rope_scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'