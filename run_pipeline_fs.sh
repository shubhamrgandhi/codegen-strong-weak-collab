#!/bin/bash

# Default values from the original script
DEFAULT_MODEL="gpt-4o-mini-2024-07-18"
# DEFAULT_MODEL="o3-mini-2025-01-31"
DEFAULT_REPO_NAME="astropy"

# Initialize variables with default values
MODEL=$DEFAULT_MODEL
REPO_NAME=$DEFAULT_REPO_NAME

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --repo_name)
      REPO_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--model MODEL] [--repo_name REPO_NAME]"
      exit 1
      ;;
  esac
done

FS_MODE="similarity_successful"
K="1"
EXP=fs_${FS_MODE}_${K}

# Set token limits based on model
if [[ "$MODEL" == "gpt-4o-mini-2024-07-18"* ]]; then
  MAX_COMPLETION_TOKENS=15000
  MAX_INPUT_TOKENS=110000
else
  MAX_COMPLETION_TOKENS=78000
  MAX_INPUT_TOKENS=118000
fi

# Run the python script with the provided or default arguments
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results/${EXP}_${MODEL} \
        --loc_file results/${EXP}_${MODEL}/retrieval.jsonl \
        --temp 0 \
        --model $MODEL \
        --max_completion_tokens $MAX_COMPLETION_TOKENS \
        --max_input_tokens $MAX_INPUT_TOKENS \
        --backend openai \
        --num_threads 8 \
        --max_retries 10 \
        --max_files 5 \
        --fs_k $K \
        --use_fs $FS_MODE


echo "sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &"

# sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &

# =======================================================================

FS_MODE="random_successful"
K="1"
EXP=fs_${FS_MODE}_${K}

# Set token limits based on model
if [[ "$MODEL" == "gpt-4o-mini-2024-07-18"* ]]; then
  MAX_COMPLETION_TOKENS=15000
  MAX_INPUT_TOKENS=110000
else
  MAX_COMPLETION_TOKENS=78000
  MAX_INPUT_TOKENS=118000
fi

# Run the python script with the provided or default arguments
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results/${EXP}_${MODEL} \
        --loc_file results/${EXP}_${MODEL}/retrieval.jsonl \
        --temp 0 \
        --model $MODEL \
        --max_completion_tokens $MAX_COMPLETION_TOKENS \
        --max_input_tokens $MAX_INPUT_TOKENS \
        --backend openai \
        --num_threads 8 \
        --max_retries 10 \
        --max_files 5 \
        --fs_k $K \
        --use_fs $FS_MODE


echo "sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &"

# sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &


# =======================================================================

FS_MODE="random_successful"
K="5"
EXP=fs_${FS_MODE}_${K}

# Set token limits based on model
if [[ "$MODEL" == "gpt-4o-mini-2024-07-18"* ]]; then
  MAX_COMPLETION_TOKENS=15000
  MAX_INPUT_TOKENS=110000
else
  MAX_COMPLETION_TOKENS=78000
  MAX_INPUT_TOKENS=118000
fi

# Run the python script with the provided or default arguments
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results/${EXP}_${MODEL} \
        --loc_file results/${EXP}_${MODEL}/retrieval.jsonl \
        --temp 0 \
        --model $MODEL \
        --max_completion_tokens $MAX_COMPLETION_TOKENS \
        --max_input_tokens $MAX_INPUT_TOKENS \
        --backend openai \
        --num_threads 8 \
        --max_retries 10 \
        --max_files 5 \
        --fs_k $K \
        --use_fs $FS_MODE


echo "sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &"

# sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &


# =======================================================================

FS_MODE="similarity_successful"
K="5"
EXP=fs_${FS_MODE}_${K}

# Set token limits based on model
if [[ "$MODEL" == "gpt-4o-mini-2024-07-18"* ]]; then
  MAX_COMPLETION_TOKENS=15000
  MAX_INPUT_TOKENS=110000
else
  MAX_COMPLETION_TOKENS=78000
  MAX_INPUT_TOKENS=118000
fi

# Run the python script with the provided or default arguments
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results/${EXP}_${MODEL} \
        --loc_file results/${EXP}_${MODEL}/retrieval.jsonl \
        --temp 0 \
        --model $MODEL \
        --max_completion_tokens $MAX_COMPLETION_TOKENS \
        --max_input_tokens $MAX_INPUT_TOKENS \
        --backend openai \
        --num_threads 8 \
        --max_retries 10 \
        --max_files 5 \
        --fs_k $K \
        --use_fs $FS_MODE


echo "sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &"

# sb-cli submit --predictions_path results/${EXP}_${MODEL}/all_preds.jsonl --run_id agentless_lite_${EXP}_${MODEL}_1 swe-bench_lite test  > run_logs/eval_agentless_lite_${EXP}_${MODEL}_1.log 2>&1 &