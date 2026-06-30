# Checkpoints

Model weights are **not** stored in this Git repository (each file exceeds GitHub's
100 MB limit). Download them from the release and place them here:

```
checkpoints/
├── sr_lora/pytorch_lora_weights_v2.safetensors   # base SR LoRA (OminiControl), ~885 MB
├── dpo_lora/adapter_model.safetensors            # ASASR AS-DPO LoRA (inference), ~111 MB
└── adv_lora/adapter_model.safetensors            # AMG adversary LoRA (training only), ~111 MB
```

- `sr_lora` + `dpo_lora` are required for **inference**.
- `adv_lora` is required only for **Stage-2 AS-DPO training** (you can also train your
  own adversary with `scripts/train_adversary.sh`).

The `adapter_config.json` files are kept in git so the directory layout and LoRA
configuration are available right after cloning.
