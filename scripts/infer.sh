#!/bin/bash
# ASASR inference (FLUX + dual LoRA).
# Usage: bash scripts/infer.sh
set -e

cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
# Disable proxies to avoid local weight loading being intercepted
export http_proxy= https_proxy= ALL_PROXY=

INPUT_DIR=${INPUT_DIR:-./examples/lr}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/inference}
SR_LORA=${SR_LORA:-./checkpoints/sr_lora/pytorch_lora_weights_v2.safetensors}
DPO_LORA=${DPO_LORA:-./checkpoints/dpo_lora/adapter_model.safetensors}
NUM_GPUS=${NUM_GPUS:-1}
SR_SCALE=${SR_SCALE:-1.0}
DPO_SCALE=${DPO_SCALE:-1.0}

python inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sr_lora_path "$SR_LORA" \
    --dpo_lora_path "$DPO_LORA" \
    --sr_scale "$SR_SCALE" \
    --dpo_scale "$DPO_SCALE" \
    --num_gpus "$NUM_GPUS" \
    --save_pair
