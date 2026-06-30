#!/bin/bash
# ASASR Stage-1: train the adversary network (AMG).
#
# The adversary network learns to imitate the reconstruction artifacts of baseline
# methods via SFT, in order to online-synthesize hard negatives for the Stage-2 AS-DPO.
# In the dataset, jpg_0 should be the baseline SR output (Real-ESRGAN/SeeSR/SUPSR),
# i.e. an artifact proxy, and jpg_1 the corresponding LQ. Pack with tools/build_dataset.py:
#   lr/  = LQ              ->  jpg_1
#   gt/  = baseline output ->  jpg_0   (note: put the "baseline output" here, not the real GT)
#
# Usage:
#   ADV_DATASET=./data/adv_dataset bash scripts/train_adversary.sh
set -e
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

MODEL_NAME=${FLUX_MODEL_PATH:-black-forest-labs/FLUX.1-dev}
ADV_DATASET=${ADV_DATASET:-./data/adv_dataset}
SR_LORA=${SR_LORA:-./checkpoints/sr_lora/pytorch_lora_weights_v2.safetensors}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/adv_$(date +%m%d_%H%M)}

NUM_GPUS=${NUM_GPUS:-8}
BATCH=${BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-8}
STEPS=${STEPS:-1000}
LR=${LR:-5e-5}                  # paper: adversary network AdamW lr=5e-5
RANK=${RANK:-16}               # paper main-setting capacity (16,16)
RESOLUTION=${RESOLUTION:-512}

mkdir -p "$OUTPUT_DIR"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --main_process_port=0 \
    --mixed_precision=bf16 \
    train_adversary.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$ADV_DATASET \
    --lora_path=$SR_LORA \
    --output_dir=$OUTPUT_DIR \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --max_train_steps=$STEPS \
    --learning_rate=$LR \
    --rank=$RANK \
    --gradient_checkpointing \
    2>&1 | tee "$OUTPUT_DIR/train_adv_$(date +%Y%m%d_%H%M%S).log"

echo "Adversary training complete -> $OUTPUT_DIR/final_adv_lora/adapter_model.safetensors"
echo "Next step: pass it as ADV_LORA to scripts/train.sh to enable AS-DPO."
