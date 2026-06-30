import os
import glob
import argparse
import tempfile
import torch
import pyiqa
from PIL import Image
import numpy as np
from collections import defaultdict


def _maybe_resize_sr_to_gt(sr_path: str, gt_path: str) -> str:
    """Return sr_path, or a temp file resized to GT WxH if sizes differ (for FR metrics)."""
    gt = Image.open(gt_path).convert("RGB")
    sr = Image.open(sr_path).convert("RGB")
    if sr.size == gt.size:
        return sr_path
    sr = sr.resize(gt.size, Image.Resampling.LANCZOS)
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    sr.save(tmp)
    return tmp

def find_matching_pairs(sr_dir, gt_dir):
    """Match SR and GT images by stem name (case-insensitive extension)."""
    sr_files = {}
    for f in sorted(os.listdir(sr_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            stem = os.path.splitext(f)[0]
            sr_files[stem] = os.path.join(sr_dir, f)

    gt_files = {}
    for f in sorted(os.listdir(gt_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            stem = os.path.splitext(f)[0]
            gt_files[stem] = os.path.join(gt_dir, f)

    pairs = []
    for stem in sorted(sr_files.keys()):
        if stem in gt_files:
            pairs.append((sr_files[stem], gt_files[stem]))
    return pairs


def evaluate(sr_dir, gt_dir, device='cuda'):
    pairs = find_matching_pairs(sr_dir, gt_dir)
    print(f"Found {len(pairs)} matched image pairs")
    if len(pairs) == 0:
        print("ERROR: No matched pairs found!")
        return {}

    # Consistent with the paper: PSNR / SSIM are computed on the Y channel of YCbCr
    # Full-reference metrics (require GT): PSNR, SSIM, LPIPS down, DISTS down
    fr_metrics = {
        'psnr': pyiqa.create_metric('psnr', device=device, test_y_channel=True, color_space='ycbcr'),
        'ssim': pyiqa.create_metric('ssim', device=device, test_y_channel=True, color_space='ycbcr'),
        'lpips': pyiqa.create_metric('lpips', device=device),
        'dists': pyiqa.create_metric('dists', device=device),
    }
    # No-reference metrics: MANIQA up, MUSIQ up, CLIPIQA+ up
    nr_metrics = {
        'maniqa': pyiqa.create_metric('maniqa', device=device),
        'musiq': pyiqa.create_metric('musiq', device=device),
        'clipiqa+': pyiqa.create_metric('clipiqa+', device=device),
    }

    results = defaultdict(list)

    for i, (sr_path, gt_path) in enumerate(pairs):
        stem = os.path.splitext(os.path.basename(sr_path))[0]
        sr_for_fr = _maybe_resize_sr_to_gt(sr_path, gt_path)

        for name, metric in fr_metrics.items():
            score = metric(sr_for_fr, gt_path).item()
            results[name].append(score)

        if sr_for_fr != sr_path:
            try:
                os.remove(sr_for_fr)
            except OSError:
                pass

        for name, metric in nr_metrics.items():
            score = metric(sr_path).item()
            results[name].append(score)

        if (i + 1) % 20 == 0 or i == len(pairs) - 1:
            print(f"  [{i+1}/{len(pairs)}] processed")

    summary = {}
    for name, scores in results.items():
        avg = np.mean(scores)
        summary[name] = avg
        print(f"  {name}: {avg:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--sr_dirs', type=str, nargs='+', required=True,
                        help='space-separated list of name:path pairs, e.g. seesr:/path/to/results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    all_results = {}
    for entry in args.sr_dirs:
        name, path = entry.split(':', 1)
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"  SR dir: {path}")
        print(f"  GT dir: {args.gt_dir}")
        print(f"{'='*60}")
        all_results[name] = evaluate(path, args.gt_dir, device=args.device)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    # Metric order and direction matching the paper's main table (up = higher is better / down = lower is better)
    metric_order = [
        ('psnr', 'PSNR↑'), ('ssim', 'SSIM↑'), ('lpips', 'LPIPS↓'), ('dists', 'DISTS↓'),
        ('maniqa', 'MANIQA↑'), ('musiq', 'MUSIQ↑'), ('clipiqa+', 'CLIPIQA+↑'),
    ]
    header = f"{'Method':<15}" + "".join(f"{lbl:>10}" for _, lbl in metric_order)
    print(header)
    print("-" * len(header))
    for name, metrics in all_results.items():
        row = f"{name:<15}"
        for key, _ in metric_order:
            v = metrics.get(key, None)
            row += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
        print(row)


if __name__ == '__main__':
    main()
