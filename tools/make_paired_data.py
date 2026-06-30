'''
 * DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation
 * Modified from SeeSR by Yuang Ai
 * 13/10/2024
'''


import os
import sys
import cv2

# Optional: point DREAMCLEAR_ROOT to a custom DreamClear/SeeSR checkout that
# bundles a tuned basicsr. Otherwise the system-installed `basicsr` is used.
_DREAMCLEAR_ROOT = os.environ.get("DREAMCLEAR_ROOT", "")
if _DREAMCLEAR_ROOT and os.path.isdir(_DREAMCLEAR_ROOT):
    sys.path.insert(0, os.path.abspath(_DREAMCLEAR_ROOT))

import torch
import torch.nn.functional as F

import argparse
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from multiprocessing import Pool
import numpy as np
import random

from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # Numpy random seed
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # current GPU random seed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gt_path",
    nargs="+",
    default=["./data/hr"],
    help="High-resolution source image directories (one or more)",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./data/paired_train",
    help="Output root directory, containing gt/ and lr/",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=192,
    help="A smaller batch makes degradation more random but slower",
)
parser.add_argument("--epoch", type=int, default=1, help="Number of generation epochs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--pool_workers",
    type=int,
    default=16,
    help="Number of cv2 disk-writing processes",
)
parser.add_argument(
    "--deg-jpeg",
    type=float,
    default=None,
    help="Override JPEG/re-compression strength knob DEG_JPEG (larger means heavier compression, about 0.55~1.15)",
)
parser.add_argument(
    "--deg-blur",
    type=float,
    default=None,
    help="Override optical blur knob DEG_BLUR (about 0.55~1.15)",
)
parser.add_argument(
    "--deg-noise",
    type=float,
    default=None,
    help="Override sensor noise knob DEG_NOISE (about 0.55~1.15)",
)
parser.add_argument(
    "--deg-resize",
    type=float,
    default=None,
    help="Override random resize / interpolation artifact knob DEG_RESIZE_STRESS (about 0.55~1.15)",
)
parser.add_argument(
    "--deg-ringing",
    type=float,
    default=None,
    help="Override sinc ringing knob DEG_RINGING; for real-world scenes <=0.55 is recommended (about 0.2~0.75)",
)
parser.add_argument(
    "--deg-second",
    type=float,
    default=None,
    help="Override second-stage degradation overall strength DEG_SECOND_STAGE (about 0.45~1.0)",
)
args = parser.parse_args()

args_training_dataset = {}

# Please set your gt path here. If you have multi dirs, you can set it as ['PATH1', 'PATH2', 'PATH3', ...]
args_training_dataset['gt_path'] = args.gt_path
args_training_dataset['dataroot_gt'] = args.gt_path
#################### REALESRGAN SETTING ###########################
############ We use the same setting as SeeSR ###########################
args_training_dataset['queue_size'] = 160
args_training_dataset['crop_size'] =  512
args_training_dataset['io_backend'] = {}
args_training_dataset['io_backend']['type'] = 'disk'

# ---------------------------------------------------------------------------
# Real-world oriented, with per-knob strength control (relative to 1.0: >1 stronger, <1 weaker):
#   - More "blurry photo" look      -> slightly increase DEG_BLUR / DEG_SECOND_STAGE
#   - More "heavy compression/repost" -> slightly increase DEG_JPEG (lowers the JPEG quality range)
#   - Cleaner sensor                 -> decrease DEG_NOISE
#   - Fewer color speckles           -> decrease DEG_NOISE and raise GRAY_NOISE_CHROMA_SUPPRESS below
#   - More resize artifacts          -> increase DEG_RESIZE_STRESS
#   - Occasional sharp ringing       -> increase DEG_RINGING (still keep <=0.6; strong sinc is rare in real photos)
# ---------------------------------------------------------------------------
# Defaults use the strongest level (all knobs = 1.0), aligned with the paper's
# "Real-ESRGAN degradation aligned with SeeSR/DreamClear". For lighter degradation
# (e.g. demos only), use the --deg_* arguments to lower them (0~1).
DEG_JPEG = 1.0 if args.deg_jpeg is None else args.deg_jpeg
DEG_BLUR = 1.0 if args.deg_blur is None else args.deg_blur
DEG_NOISE = 1.0 if args.deg_noise is None else args.deg_noise
DEG_RESIZE_STRESS = 1.0 if args.deg_resize is None else args.deg_resize
DEG_RINGING = 1.0 if args.deg_ringing is None else args.deg_ringing
DEG_SECOND_STAGE = 1.0 if args.deg_second is None else args.deg_second
# Chroma noise suppression: a higher gray_prob biases Gaussian/Poisson noise toward
# monochrome, reducing color speckles caused by independent per-RGB-channel noise.
GRAY_NOISE_CHROMA_SUPPRESS = 0.48

def _lerp(a, b, t):
    return a + (b - a) * t

# Dataset kernels: slightly higher isotropy (more like optical defocus); sinc scaled by DEG_RINGING
_kp = [0.42, 0.26, 0.14, 0.05, 0.09, 0.04]
args_training_dataset['blur_kernel_size'] = max(5, int(round(_lerp(13, 21, DEG_BLUR))))
args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob'] = _kp
args_training_dataset['sinc_prob'] = 0.06 * DEG_RINGING
args_training_dataset['blur_sigma'] = [0.2, _lerp(1.15, 2.35, DEG_BLUR)]
args_training_dataset['betag_range'] = [0.45, _lerp(2.0, 3.0, DEG_BLUR)]
args_training_dataset['betap_range'] = [1, _lerp(1.55, 2.0, DEG_BLUR)]

args_training_dataset['blur_kernel_size2'] = max(5, int(round(_lerp(9, 15, DEG_BLUR * DEG_SECOND_STAGE))))
args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob2'] = _kp
args_training_dataset['sinc_prob2'] = 0.05 * DEG_RINGING
args_training_dataset['blur_sigma2'] = [0.15, _lerp(0.85, 1.75, DEG_BLUR * DEG_SECOND_STAGE)]
args_training_dataset['betag_range2'] = [0.45, _lerp(2.0, 3.0, DEG_BLUR)]
args_training_dataset['betap_range2'] = [1, _lerp(1.55, 2.0, DEG_BLUR)]

args_training_dataset['final_sinc_prob'] = _lerp(0.32, 0.78, DEG_RINGING)

args_training_dataset['use_hflip'] = False
args_training_dataset['use_rot'] = False

seed_ = args.seed
set_seed(seed_)

train_dataset = RealESRGANDataset(args_training_dataset)
batch_size = args.batch_size
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=16,
    drop_last=False,
)

#################### REAL-ESRGAN: real-world oriented, overall lighter (tuned via DEG_* above) ####
# Baseline: typical phone/social images - moderately high JPEG, mild blur, light noise,
# conservative resize, weaker second stage.
args_degradation = {}

# First stage: higher keep ratio; resize avoids extreme downscaling
_p_up = 0.28 * DEG_RESIZE_STRESS
_p_dn = 0.38 * DEG_RESIZE_STRESS
args_degradation['resize_prob'] = [_p_up, _p_dn, 1.0 - _p_up - _p_dn]
args_degradation['resize_range'] = [
    _lerp(0.55, 0.28, DEG_RESIZE_STRESS),
    _lerp(1.25, 1.65, DEG_RESIZE_STRESS),
]
args_degradation['gaussian_noise_prob'] = _lerp(0.03, 0.16, DEG_NOISE)
args_degradation['noise_range'] = [0.5, _lerp(1.5, 5.5, DEG_NOISE)]
args_degradation['poisson_scale_range'] = [0.01, _lerp(0.08, 0.5, DEG_NOISE)]
args_degradation['gray_noise_prob'] = min(
    0.88,
    _lerp(0.18, 0.42, DEG_NOISE) + GRAY_NOISE_CHROMA_SUPPRESS,
)
# JPEG: a larger DEG_JPEG lowers the quality floor (heavier compression), simulating repeated saves/reposts
_j1_lo = _lerp(74, 48, DEG_JPEG)
_j1_hi = _lerp(96, 88, DEG_JPEG)
args_degradation['jpeg_range'] = [_j1_lo, max(_j1_lo + 4, _j1_hi)]

# Second stage: overall weaker, more like "saving the image once more" than a second heavy degradation chain
args_degradation['second_blur_prob'] = _lerp(0.22, 0.55, DEG_SECOND_STAGE)
_p2_up = 0.18 * DEG_RESIZE_STRESS * DEG_SECOND_STAGE
_p2_dn = 0.28 * DEG_RESIZE_STRESS * DEG_SECOND_STAGE
args_degradation['resize_prob2'] = [_p2_up, _p2_dn, 1.0 - _p2_up - _p2_dn]
args_degradation['resize_range2'] = [
    _lerp(0.68, 0.48, DEG_RESIZE_STRESS),
    _lerp(1.12, 1.32, DEG_RESIZE_STRESS),
]
args_degradation['gaussian_noise_prob2'] = _lerp(0.02, 0.10, DEG_NOISE * DEG_SECOND_STAGE)
args_degradation['noise_range2'] = [0.5, _lerp(1.2, 4.0, DEG_NOISE * DEG_SECOND_STAGE)]
args_degradation['poisson_scale_range2'] = [0.01, _lerp(0.06, 0.32, DEG_NOISE * DEG_SECOND_STAGE)]
args_degradation['gray_noise_prob2'] = min(
    0.88,
    _lerp(0.15, 0.38, DEG_NOISE) + GRAY_NOISE_CHROMA_SUPPRESS * 0.95,
)
_j2_lo = _lerp(78, 52, DEG_JPEG)
_j2_hi = _lerp(97, 90, DEG_JPEG)
args_degradation['jpeg_range2'] = [_j2_lo, max(_j2_lo + 3, _j2_hi)]

args_degradation['gt_size'] = 512
# Stronger second stage -> fewer "resize + JPEG only" light paths; the cap is raised a bit so the overall result is cleaner
args_degradation["no_degradation_prob"] = min(
    0.55, 0.20 + (1.0 - DEG_SECOND_STAGE) * 0.30
)


from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
import random
import torch.nn.functional as F

def realesrgan_degradation(batch,  args_degradation, use_usm=True, sf=4, resize_lq=True):
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # Small fraction of samples: only "downscale + a single JPEG", close to screenshots / simple resize, with no blur chain
    if random.random() < float(args_degradation.get("no_degradation_prob", 0.0)):
        out = im_gt
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(ori_h // sf, ori_w // sf), mode=mode)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation["jpeg_range2"])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        im_lq = torch.clamp(out, 0, 1.0)
        gt_size = args_degradation["gt_size"]
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, sf)
        im_gt = torch.clamp(im_gt, 0, 1)
        im_lq = torch.clamp(im_lq, 0, 1)
        return im_lq, im_gt

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            args_degradation['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, args_degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(args_degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob']
    if random.random() < args_degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < args_degradation['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            args_degradation['resize_prob2'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, args_degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(args_degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(int(ori_h / sf * scale),
                    int(ori_w / sf * scale)),
            mode=mode,
            )
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob2']
    if random.random() < args_degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf,
                        ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf,
                        ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # random crop
    gt_size = args_degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, sf)
    lq, gt = im_lq, im_gt


    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)

    return lq, gt


root_path = args.save_dir
gt_path = os.path.join(root_path, 'gt')
lr_path = os.path.join(root_path, 'lr')
os.makedirs(gt_path, exist_ok=True)
os.makedirs(lr_path, exist_ok=True)

def save_images(filename_, lr_, gt_):
    """Save images, preserving the original filename (extension changed to .png)."""
    base_name = os.path.splitext(filename_)[0] + '.png'

    lr_save_path = os.path.join(lr_path, base_name)
    gt_save_path = os.path.join(gt_path, base_name)

    cv2.imwrite(lr_save_path, 255 * lr_.squeeze().permute(1, 2, 0).numpy()[..., ::-1])
    cv2.imwrite(gt_save_path, 255 * gt_.squeeze().permute(1, 2, 0).numpy()[..., ::-1])
    del lr_, gt_

epochs = args.epoch
step = 0
with torch.no_grad():
    pool = Pool(processes=max(1, args.pool_workers))
    for epoch in range(epochs):
        for num_batch, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Generating training samples"):
            lr_batch, gt_batch = realesrgan_degradation(batch, args_degradation=args_degradation)

            lr_batch = lr_batch.detach().cpu()
            gt_batch = gt_batch.detach().cpu()

            # Original filename list
            gt_paths = batch['gt_path']

            for i in range(lr_batch.shape[0]):
                step += 1
                lr = lr_batch[i, ...]
                gt = gt_batch[i, ...]
                # Use the original filename
                original_filename = os.path.basename(gt_paths[i])
                pool.apply_async(save_images, args=(original_filename, lr, gt))

            del lr_batch, gt_batch
            torch.cuda.empty_cache()

    pool.close()
    pool.join()