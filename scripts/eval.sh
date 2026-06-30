#!/bin/bash
# ASASR evaluation (pyiqa: PSNR / SSIM / MANIQA / CLIPIQA+).
# Usage: bash scripts/eval.sh
set -e

cd "$(dirname "$0")/.."

GT_DIR=${GT_DIR:-./examples/gt}
SR_DIR=${SR_DIR:-./outputs/inference}
TAG=${TAG:-asasr}

python eval/eval_pyiqa.py \
    --gt_dir "$GT_DIR" \
    --sr_dirs "${TAG}:${SR_DIR}"
