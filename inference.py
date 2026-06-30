"""
ASASR inference: FLUX + dual LoRA (SR + DPO) super-resolution.

Usage:
    python inference.py --input_dir examples/lr --output_dir outputs/

Set FLUX_MODEL_PATH env var to your local FLUX.1-dev checkpoint dir if you
do not want to download from HF Hub.
"""
import argparse
import os
import time

import torch
import torch.multiprocessing as mp
from diffusers.pipelines import FluxPipeline
from PIL import Image
from tqdm import tqdm

from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything
from tools.color_fix import adain_color_fix


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_DIR = os.path.join(REPO_ROOT, "examples", "lr")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "inference")
DEFAULT_FLUX_MODEL = os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
DEFAULT_SR_LORA = os.path.join(
    REPO_ROOT, "checkpoints", "sr_lora", "pytorch_lora_weights_v2.safetensors"
)
DEFAULT_DPO_LORA = os.path.join(
    REPO_ROOT, "checkpoints", "dpo_lora", "adapter_model.safetensors"
)


def load_pipeline(gpu_id, flux_path, sr_lora_path, dpo_lora_path, sr_scale, dpo_scale):
    device = f"cuda:{gpu_id}"
    pipe = FluxPipeline.from_pretrained(
        flux_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    # Load the local LoRA file via (directory, weight_name) to avoid diffusers treating
    # the full file path as a repo id and querying the Hub under HF_HUB_OFFLINE=1, which
    # raises "When using the offline mode, you must specify a `weight_name`."
    sr_dir, sr_name = os.path.split(sr_lora_path)
    dpo_dir, dpo_name = os.path.split(dpo_lora_path)
    pipe.load_lora_weights(sr_dir, weight_name=sr_name, adapter_name="sr")
    pipe.load_lora_weights(dpo_dir, weight_name=dpo_name, adapter_name="dpo")
    pipe.set_adapters(["sr", "dpo"], adapter_weights=[sr_scale, dpo_scale])
    return pipe


def _save_pair(left: Image.Image, right: Image.Image, save_path: str):
    left = left.convert("RGB")
    right = right.convert("RGB")
    h = max(left.height, right.height)
    if left.height != h:
        nw = max(1, int(round(left.width * h / left.height)))
        left = left.resize((nw, h), Image.BICUBIC)
    if right.height != h:
        nw = max(1, int(round(right.width * h / right.height)))
        right = right.resize((nw, h), Image.BICUBIC)
    pair = Image.new("RGB", (left.width + right.width, h))
    pair.paste(left, (0, 0))
    pair.paste(right, (left.width, 0))
    pair.save(save_path)


def process_images(gpu_id, image_list, args):
    if len(image_list) == 0:
        return

    print(f"[GPU {gpu_id}] loading pipeline ...")
    pipe = load_pipeline(
        gpu_id,
        args.flux_path,
        args.sr_lora_path,
        args.dpo_lora_path,
        args.sr_scale,
        args.dpo_scale,
    )
    print(f"[GPU {gpu_id}] ready, processing {len(image_list)} images")

    pbar = tqdm(image_list, desc=f"GPU {gpu_id}", position=gpu_id, ncols=100)
    for filename in pbar:
        image_path = os.path.join(args.input_dir, filename)
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        min_dim = min(w, h)
        image = image.crop(
            ((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2)
        ).resize((args.resolution, args.resolution), Image.BICUBIC)

        condition = Condition("sr", image)
        seed_everything()

        result_img = generate(
            pipe,
            prompt="",
            conditions=[condition],
            default_lora=True,
        ).images[0]

        result_img = adain_color_fix(result_img, image)
        out_path = os.path.join(args.output_dir, filename)
        result_img.save(out_path)
        if args.save_pair:
            stem, _ = os.path.splitext(filename)
            _save_pair(image, result_img, os.path.join(args.output_dir, f"{stem}_pair.png"))

    print(f"[GPU {gpu_id}] done.")


def main():
    parser = argparse.ArgumentParser(description="ASASR FLUX dual-LoRA SR inference")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flux_path", type=str, default=DEFAULT_FLUX_MODEL,
                        help="FLUX.1-dev model path (local dir or HF repo id)")
    parser.add_argument("--sr_lora_path", type=str, default=DEFAULT_SR_LORA)
    parser.add_argument("--dpo_lora_path", type=str, default=DEFAULT_DPO_LORA)
    parser.add_argument("--sr_scale", type=float, default=1.0)
    parser.add_argument("--dpo_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--save_pair", action="store_true",
                        help="also save side-by-side input|output png")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_images = sorted(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    )
    total = len(all_images)

    print("=" * 60)
    print(f"input_dir : {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"images    : {total}")
    print(f"num_gpus  : {args.num_gpus}")
    print(f"sr_lora   : {args.sr_lora_path}  (scale={args.sr_scale})")
    print(f"dpo_lora  : {args.dpo_lora_path} (scale={args.dpo_scale})")
    print("=" * 60)

    if total == 0:
        print("no images to process.")
        return

    chunks = [[] for _ in range(args.num_gpus)]
    for i, img in enumerate(all_images):
        chunks[i % args.num_gpus].append(img)

    start = time.time()

    if args.num_gpus == 1:
        process_images(0, chunks[0], args)
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_id in range(args.num_gpus):
            p = mp.Process(target=process_images, args=(gpu_id, chunks[gpu_id], args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    elapsed = time.time() - start
    print("=" * 60)
    print(f"done. {total} images in {elapsed:.1f}s "
          f"({elapsed / total:.2f}s/img)  → {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
