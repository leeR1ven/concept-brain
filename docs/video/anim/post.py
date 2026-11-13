"""Turn scene-linear EXR frames into a video: bloom, filmic tone curve, dither (no banding in the dark fog), H.264.
usage: python3 post.py <exrdir> <out.mp4> <fps> [--stills dir every]"""
import glob
import os
import subprocess
import sys

import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def read(p):
    with OpenEXR.File(p) as f:
        c = f.channels()
        if 'RGB' in c:
            a = c['RGB'].pixels
        else:
            a = np.stack([c['R'].pixels, c['G'].pixels, c['B'].pixels], -1)
    return torch.as_tensor(a.astype(np.float32), device=DEV).permute(2, 0, 1)[None]


def blur(x, sigma):
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum())[None, None]
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), k[:, :, None, :].repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k[:, :, :, None].repeat(c, 1, 1, 1), groups=c)


def bloom(x):
    """soft glow from the bright parts, at several sizes"""
    out, cur = 0.0, torch.clamp(x - 0.75, min=0.0)
    h, w = x.shape[2:]
    for lv, gain in ((1, 0.16), (2, 0.12), (4, 0.09), (8, 0.06), (16, 0.04)):
        small = F.avg_pool2d(cur, lv) if lv > 1 else cur
        g = blur(small, 3.0)
        out = out + gain * F.interpolate(g, (h, w), mode='bilinear', align_corners=False)
    return x + out


def tone(x):
    """ACES fit, then display gamma"""
    x = x * 0.75
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return torch.clamp(y, 0, 1) ** (1 / 2.2)


def main():
    src, out, fps = sys.argv[1], sys.argv[2], sys.argv[3]
    fs = sorted(glob.glob(f'{src}/*.exr'))
    h, w = read(fs[0]).shape[2:]
    still_dir = sys.argv[5] if len(sys.argv) > 5 else None
    every = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                           '-r', fps, '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt',
                           'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    g = torch.Generator(device=DEV).manual_seed(5)
    for i, p in enumerate(fs):
        fade = min(1.0, i / 30, (len(fs) - 1 - i) / 27)  # one second in, just under a second out
        y = tone(bloom(read(p)) * fade * fade * (3 - 2 * fade))
        y = y + (torch.rand(y.shape, generator=g, device=DEV) - torch.rand(y.shape, generator=g, device=DEV)) / 255.0
        b = (torch.clamp(y, 0, 1) * 255 + 0.5).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        ff.stdin.write(b.tobytes())
        if still_dir and i % every == 0:
            os.makedirs(still_dir, exist_ok=True)
            with open(f'{still_dir}/{os.path.basename(p)[:-4]}.ppm', 'wb') as f:
                f.write(f'P6 {w} {h} 255\n'.encode() + b.tobytes())
    ff.stdin.close()
    ff.wait()
    print('POST', out, len(fs), 'frames', round(os.path.getsize(out) / 1e6, 1), 'MB', flush=True)


if __name__ == '__main__':
    main()
