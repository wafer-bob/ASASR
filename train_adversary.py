"""
ASASR Stage-1: train the adversary network (Adversarial Manifold Guidance, AMG).

The adversary network A_phi is trained via supervised fine-tuning (SFT) to imitate
the reconstruction distortions of baseline methods, so that in the second stage
(AS-DPO) it can synthesize hard negatives semantically aligned with the winner online.

This script implements that SFT stage: on top of a frozen FLUX + SR-LoRA (base_sft),
a trainable LoRA adapter (adv) is added and trained with a flow-matching MSE loss
(minimizing the Euclidean residual energy J_{L^2}, corresponding to Prop.1/Prop.2 of
the paper) so that it learns to reproduce the given target images.

Dataset format (HuggingFace Dataset, consistent with tools/build_dataset.py):
    jpg_0 = adversarial target (recommended: output of a baseline SR method such as
            Real-ESRGAN/SeeSR/SUPSR, i.e. an artifact proxy)
    jpg_1 = LQ low-quality input (condition)
If jpg_0 uses the GT, this degenerates to plain SR-SFT; the paper uses baseline
outputs to learn the real artifact manifold.

The resulting checkpoints/.../adapter_model.safetensors can be passed directly as
train.py's --adv_lora_path.

Usage (see scripts/train_adversary.sh):
    accelerate launch --num_processes=8 --mixed_precision=bf16 train_adversary.py \
        --pretrained_model_name_or_path=<FLUX> --dataset_name=<adv_ds> \
        --lora_path=<SR_LoRA> --output_dir=./outputs/adv --learning_rate=5e-5
"""
import argparse
import logging
import math
import os

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_from_disk, load_dataset
from peft import LoraConfig, get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling

from src.flux.pipeline_tools import encode_images
from src.flux.transformer import tranformer_forward
from src.flux.condition import Condition

logger = get_logger(__name__)

# target_modules identical to the SR/DPO LoRA, so the saved adapter can be loaded by load_lora_weights
TARGET_MODULES = (
    "(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear"
    "|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k"
    "|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q"
    "|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v"
    "|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0"
    "|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2"
    "|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear"
    "|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp"
    "|.*single_transformer_blocks\\.[0-9]+\\.proj_out"
    "|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k"
    "|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q"
    "|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v"
    "|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"
)

MODEL_CONFIG = {"union_cond_attn": True, "add_cond_attn": False, "latent_lora": False}


def pack_latents(latents):
    """[B,C,H,W] -> ([B, (H//2)*(W//2), C*4], ids[B,seq,3]); ids use the (0,h,w) convention (official Flux)."""
    B, C, H, W = latents.shape
    assert H % 2 == 0 and W % 2 == 0
    lat = latents.view(B, C, H // 2, 2, W // 2, 2)
    lat = lat.permute(0, 2, 4, 1, 3, 5).contiguous()
    packed = lat.view(B, (H // 2) * (W // 2), C * 4)
    h_idx = torch.arange(H // 2, device=latents.device, dtype=torch.long)
    w_idx = torch.arange(W // 2, device=latents.device, dtype=torch.long)
    h_grid, w_grid = torch.meshgrid(h_idx, w_idx, indexing="ij")
    ids = torch.stack([torch.zeros_like(h_grid), h_grid, w_grid], dim=-1)
    ids = ids.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 3)
    return packed, ids


def unpack_latents(packed, height, width):
    """[B, seq, C*4] -> [B, C, H, W]."""
    B, seq, packed_c = packed.shape
    C = packed_c // 4
    H, W = height, width
    lat = packed.view(B, H // 2, W // 2, C, 2, 2)
    lat = lat.permute(0, 3, 1, 4, 2, 5).contiguous()
    return lat.view(B, C, H, W)


def get_sigmas(scheduler, timesteps, device, n_dim, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def parse_args():
    p = argparse.ArgumentParser(description="Train ASASR adversary (AMG) via SFT.")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True,
                   help="HF dataset (save_to_disk or imagefolder); columns jpg_0=adversarial target, jpg_1=LQ")
    p.add_argument("--lora_path", type=str, required=True, help="base SR LoRA (base_sft, frozen)")
    p.add_argument("--output_dir", type=str, default="./outputs/adv")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_train_steps", type=int, default=1000)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=5e-5, help="paper: adversary network AdamW lr=5e-5")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=16, help="adversary network capacity (paper main setting: 16)")
    p.add_argument("--checkpointing_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="bf16")
    return p.parse_args()


def save_adapter(transformer, accelerator, save_dir):
    """Save the adv adapter in a format compatible with Diffusers load_lora_weights."""
    os.makedirs(save_dir, exist_ok=True)
    full = accelerator.get_state_dict(transformer)
    unet = accelerator.unwrap_model(transformer)
    lora_sd = {k: v for k, v in full.items() if "adv" in k and "lora" in k.lower()}
    new_sd = {}
    for k, v in lora_sd.items():
        nk = k.replace("base_model.model.", "").replace(".adv.", ".")
        if not nk.startswith("transformer."):
            nk = "transformer." + nk
        new_sd[nk] = v
    unet.peft_config["adv"].save_pretrained(save_dir)
    from safetensors.torch import save_file
    save_file(new_sd, os.path.join(save_dir, "adapter_model.safetensors"))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if args.seed is not None:
        set_seed(args.seed)
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # ---- Load FLUX ----
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # ---- base_sft (SR LoRA, frozen) + adv (trainable) ----
    pipeline.load_lora_weights(
        os.path.dirname(args.lora_path), weight_name=os.path.basename(args.lora_path),
        adapter_name="base_sft",
    )
    for n, p in pipeline.transformer.named_parameters():
        if "base_sft" in n:
            p.requires_grad_(False)
    adv_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=TARGET_MODULES,
                         init_lora_weights="gaussian")
    pipeline.transformer.add_adapter(adv_cfg, adapter_name="adv")
    for n, p in pipeline.transformer.named_parameters():
        p.requires_grad_("adv" in n)
    pipeline.transformer.set_adapters(["base_sft", "adv"])
    if args.gradient_checkpointing:
        pipeline.transformer.enable_gradient_checkpointing()
    pipeline.transformer.to(accelerator.device)

    # ---- Data ----
    try:
        dataset = load_from_disk(args.dataset_name)
    except Exception:
        dataset = load_dataset("imagefolder", data_dir=args.dataset_name)["train"]
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    tfm = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        tgt = [tfm(im.convert("RGB")) for im in examples["jpg_0"]]
        lq = [tfm(im.convert("RGB")) for im in examples["jpg_1"]]
        examples["target"] = tgt
        examples["lq"] = lq
        return examples

    dataset.set_transform(preprocess)

    def collate(examples):
        return {
            "target": torch.stack([e["target"] for e in examples]).contiguous().float(),
            "lq": torch.stack([e["lq"] for e in examples]).contiguous().float(),
        }

    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=collate, batch_size=args.train_batch_size,
        num_workers=0,
    )

    params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler("constant_with_warmup", optimizer=optimizer,
                             num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                             num_training_steps=args.max_train_steps * accelerator.num_processes)

    pipeline.transformer, optimizer, loader, lr_sched = accelerator.prepare(
        pipeline.transformer, optimizer, loader, lr_sched
    )

    if accelerator.is_main_process:
        logger.info(f"Start training adversary network (AMG/SFT): rank={args.rank}, lr={args.learning_rate}, "
                    f"steps={args.max_train_steps}, samples={len(dataset)}")

    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    sr_type_id = Condition.get_type_id("sr")
    transformer = pipeline.transformer
    # Under DDP, accelerator.prepare wraps the transformer in DistributedDataParallel,
    # but tranformer_forward directly accesses submodules like .x_embedder (bypassing DDP.__call__),
    # so the forward must use the unwrapped module (gradients still flow to the same parameters).
    fwd_model = accelerator.unwrap_model(transformer)
    done = False
    while not done:
        for batch in loader:
            with accelerator.accumulate(transformer):
                target_img = batch["target"].to(accelerator.device, dtype=weight_dtype)
                lq_img = batch["lq"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = pipeline.vae.encode(target_img).latent_dist.sample()
                    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bsz,
                    logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
                idx = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[idx].to(latents.device)
                sigmas = get_sigmas(noise_scheduler, timesteps, accelerator.device,
                                    latents.ndim, latents.dtype)
                noisy = (1.0 - sigmas) * latents + sigmas * noise
                target_v = noise - latents

                with torch.no_grad():
                    cond_latents, cond_ids = encode_images(pipeline, lq_img)
                    if cond_latents.shape[0] != bsz:
                        cond_latents = cond_latents.repeat(bsz // cond_latents.shape[0], 1, 1)
                cond_type_ids = torch.full((cond_ids.shape[0], 1), fill_value=sr_type_id,
                                           device=latents.device, dtype=cond_ids.dtype)

                packed_noisy, latent_ids = pack_latents(noisy)
                seq_tx = 512
                tf_cfg = fwd_model.config
                prompt_embeds = torch.zeros(bsz, seq_tx, tf_cfg.joint_attention_dim,
                                            device=accelerator.device, dtype=weight_dtype)
                pooled = torch.zeros(bsz, tf_cfg.pooled_projection_dim,
                                     device=accelerator.device, dtype=weight_dtype)
                txt_ids = torch.zeros(bsz, seq_tx, 3, device=accelerator.device, dtype=torch.long)

                pred = tranformer_forward(
                    fwd_model, model_config=MODEL_CONFIG,
                    condition_latents=cond_latents, condition_ids=cond_ids,
                    condition_type_ids=cond_type_ids,
                    hidden_states=packed_noisy, timestep=timesteps / 1000,
                    img_ids=latent_ids, pooled_projections=pooled,
                    encoder_hidden_states=prompt_embeds, txt_ids=txt_ids,
                    return_dict=False,
                )[0]
                lh = args.resolution // 8
                pred = unpack_latents(pred, height=lh, width=lh)

                # paper: the adversary network minimizes the Euclidean residual energy J_{L^2} (standard flow-matching MSE)
                loss = F.mse_loss(pred.float(), target_v.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_adapter(transformer, accelerator,
                                 os.path.join(args.output_dir, f"checkpoint-{global_step}", "adv_lora"))
                if global_step >= args.max_train_steps:
                    done = True
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_adapter(transformer, accelerator, os.path.join(args.output_dir, "final_adv_lora"))
        logger.info(f"Adversary network saved: {os.path.join(args.output_dir, 'final_adv_lora')}/adapter_model.safetensors")
        logger.info("   Usage: pass it as train.py's --adv_lora_path to enable AS-DPO.")


if __name__ == "__main__":
    main()
