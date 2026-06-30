#!/bin/bash
# ASASR Stage-2 training: FLUX + Adversarial Sobolev-DPO (AS-DPO).
#
# This is the paper's full method (AS-DPO): on top of Sobolev frequency-weighted DPO,
# it uses a pre-trained adversary network (adversary LoRA) to online-synthesize hard
# negatives that are semantically aligned with the winner.
# Train the adversary first with scripts/train_adversary.sh (see README section 4).
#
# Usage:
#   bash scripts/train.sh                     # Full AS-DPO (requires a pre-trained ADV_LORA)
#   BETA=4000 SOBOLEV_S=1.5 bash scripts/train.sh
set -e

cd "$(dirname "$0")/.."

# ---- HF / cache ----
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# ---- paths (override as needed) ----
MODEL_NAME=${FLUX_MODEL_PATH:-black-forest-labs/FLUX.1-dev}
DATASET=${DATASET:-./data/dataset}
SR_LORA=${SR_LORA:-./checkpoints/sr_lora/pytorch_lora_weights_v2.safetensors}
# Adversary LoRA (the paper's AMG). Train it first with scripts/train_adversary.sh.
ADV_LORA=${ADV_LORA:-./checkpoints/adv_lora/adapter_model.safetensors}
ADV_STRENGTH=${ADV_STRENGTH:-0.1}      # lambda, adversarial perturbation strength (paper uses 0.1 for visualization/training)
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/train_$(date +%m%d_%H%M)}

# ---- DPO hyper-params (aligned with the paper) ----
# Note: the in-code logistic temperature is 0.5*beta_dpo (train.py: scale_term = -0.5*beta_dpo),
# i.e. the "effective beta = 0.5 * BETA". The paper uses effective beta=2000, hence BETA defaults to 4000.
BETA=${BETA:-4000}
SOBOLEV_S=${SOBOLEV_S:-1.5}            # Sobolev exponent s (paper setting 1.5)

# ---- training (aligned with the paper: 8xGPU, per-device 8, grad-accum 8 -> global batch 512) ----
NUM_GPUS=${NUM_GPUS:-8}
BATCH=${BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-8}
STEPS=${STEPS:-1000}
LR=${LR:-1e-5}                         # paper: AdamW lr=1e-5 (no longer uses --scale_lr)
RESOLUTION=${RESOLUTION:-512}          # HQ size; x4 SR (LQ = 128)
# Upper bound on the number of training samples; set to 0 to use the whole dataset (paper uses the full DIV2K+LSDIR)
MAX_SAMPLES=${MAX_SAMPLES:-0}

mkdir -p "$OUTPUT_DIR"

# Assemble optional arguments
EXTRA_ARGS=""
if [ -n "$ADV_LORA" ] && [ -f "$ADV_LORA" ]; then
    echo "Adversary (AMG) enabled: ADV_LORA=$ADV_LORA  strength=$ADV_STRENGTH"
    EXTRA_ARGS="$EXTRA_ARGS --adv_lora_path=$ADV_LORA --adv_strength=$ADV_STRENGTH"
else
    echo "Adversary network not found (ADV_LORA=$ADV_LORA)."
    echo "    For the full AS-DPO method, first run scripts/train_adversary.sh to train the adversary network."
fi
if [ "${MAX_SAMPLES}" != "0" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_train_samples=$MAX_SAMPLES"
fi

accelerate launch \
    --num_processes=$NUM_GPUS \
    --main_process_port=0 \
    --mixed_precision=bf16 \
    train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET \
    --train_batch_size=$BATCH \
    --resolution=$RESOLUTION \
    --mixed_precision=bf16 \
    --dataloader_num_workers=0 \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --max_train_steps=$STEPS \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=200 \
    --learning_rate=$LR \
    --checkpointing_steps=50 \
    --beta_dpo=$BETA \
    --sobolev_s=$SOBOLEV_S \
    --flux \
    --gradient_checkpointing \
    --lora_path=$SR_LORA \
    --output_dir=$OUTPUT_DIR \
    $EXTRA_ARGS \
    2>&1 | tee "$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"
