import os, math, argparse, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
import time

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Utils
# ----------------------------

def to_tensor_img(pil_img):
    arr = np.array(pil_img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2,0,1).float() / 255.0

def save_depth_png(depth_m: np.ndarray,
                   out_png: str,
                   mode: str = "fixed",
                   vmin: float = 0.3,
                   vmax: float = 10.0,
                   lo_p: float = 1.0,
                   hi_p: float = 99.0):
    """
    Save a depth map as an 8-bit PNG for visualization.

    mode = "fixed": clamp to [vmin, vmax] meters, then normalize.
    mode = "auto" : compute percentiles [lo_p, hi_p] per-image, then normalize.
    """
    d = depth_m.astype(np.float32).copy()
    if mode == "fixed":
        d = np.clip(d, vmin, vmax)
        d = (d - vmin) / (vmax - vmin + 1e-8)
    elif mode == "auto":
        lo, hi = np.percentile(d, [lo_p, hi_p])
        if hi <= lo:  # fallback if degenerate
            d[:] = 0.0
        else:
            d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown viz mode: {mode}")
    d8 = (d * 255.0).astype(np.uint8)
    imageio.imwrite(out_png, d8)

def read_depth_png_auto(p: Path):
    im = Image.open(p)
    arr = np.array(im)
    arr = arr.astype(np.float32)
    if arr.dtype == np.uint16 or arr.max() > 50.0:
        arr = arr / 1000.0  # mm -> m
    return arr

# ----------------------------
# Dataset
# ----------------------------

class NYUNativeTrain(Dataset):
    """
    root/nyu2_train/<scene>/*.jpg and matching *.png (same stem) for depth
    """
    def __init__(self, root, resize_hw=(240,320), center_crop=True, jitter=True):
        self.root = Path(root) / "nyu2_train"
        self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.jitter = jitter

        self.pairs = []
        scenes = [d for d in self.root.glob("*") if d.is_dir()]
        for s in tqdm(scenes, desc="Scan train scenes"):
            jpgs = sorted(s.glob("*.jpg"))
            for rgbp in jpgs:
                dep = rgbp.with_suffix(".png")
                if dep.exists():
                    self.pairs.append((rgbp, dep))

        # standard NYU crop box on 480x640
        self.crop_box = (41,45,601,471)

    def __len__(self): return len(self.pairs)

    def _apply_crop(self, rgb_pil, depth_np):
        l,t,r,b = self.crop_box
        return rgb_pil.crop((l,t,r,b)), depth_np[t:b, l:r]

    def _color_jitter(self, rgb_t):
        if np.random.rand() < 0.5:
            s = 0.9 + 0.2*np.random.rand(); rgb_t = torch.clamp(rgb_t*s, 0, 1)
        if np.random.rand() < 0.5:
            m = (np.random.rand(3)*0.1 - 0.05); rgb_t = torch.clamp(rgb_t + torch.tensor(m)[:,None,None], 0,1)
        return rgb_t

    def __getitem__(self, idx):
        rgbp, depp = self.pairs[idx]
        rgb = Image.open(rgbp).convert("RGB")
        depth = read_depth_png_auto(depp)

        if self.center_crop:
            rgb, depth = self._apply_crop(rgb, depth)

        H,W = self.resize_hw
        rgb = rgb.resize((W,H), Image.BILINEAR)
        depth = np.array(Image.fromarray(depth).resize((W,H), Image.NEAREST), dtype=np.float32)
        rgb_t = to_tensor_img(rgb)
        depth_t = torch.from_numpy(depth).float().clamp(0.3,10.0)

        if self.jitter:
            rgb_t = self._color_jitter(rgb_t)

        return rgb_t, depth_t, rgbp.name

class NYUNativeTest(Dataset):
    """
    root/nyu2_test/*_colors.png & *_depth.png
    """
    def __init__(self, root, resize_hw=(240,320), center_crop=True):
        self.root = Path(root) / "nyu2_test"
        self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.files = []

        color_files = sorted(self.root.glob("*_colors.png"))
        for c in color_files:
            d = c.with_name(c.stem.replace("_colors","_depth") + c.suffix)
            if d.exists():
                self.files.append((c, d))

        self.crop_box = (41,45,601,471)

    def __len__(self): return len(self.files)

    def _apply_crop(self, rgb_pil, depth_np):
        l,t,r,b = self.crop_box
        return rgb_pil.crop((l,t,r,b)), depth_np[t:b, l:r]

    def __getitem__(self, idx):
        rgbp, depp = self.files[idx]
        rgb = Image.open(rgbp).convert("RGB")
        depth = read_depth_png_auto(depp)
        if self.center_crop:
            rgb, depth = self._apply_crop(rgb, depth)
        H,W = self.resize_hw
        rgb = rgb.resize((W,H), Image.BILINEAR)
        depth = np.array(Image.fromarray(depth).resize((W,H), Image.NEAREST), dtype=np.float32)
        rgb_t = to_tensor_img(rgb)
        depth_t = torch.from_numpy(depth).float().clamp(0.3,10.0)
        return rgb_t, depth_t, rgbp.stem  # stem like "00000_colors"

# ----------------------------
# Defocus simulator
# ----------------------------

def make_invdepth_edges(min_m=0.5, max_m=10.0, J=12):
    inv = torch.linspace(1/max_m, 1/min_m, J+1)
    return (1.0 / inv).float()

def depth_to_soft_masks(depth_m, edges):
    B,H,W = depth_m.shape; J = edges.numel()-1
    masks = torch.zeros(B, J, 1, H, W, device=depth_m.device, dtype=depth_m.dtype)
    for j in range(J):
        m = (depth_m>=edges[j]) & (depth_m<edges[j+1])
        masks[:,j,0] = m.float()https://prod.liveshare.vsengsaas.visualstudio.com/join?BEB81EE04963DB19C61EC9E2BBF4881A0B2F
    masks = F.avg_pool3d(masks, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
    masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-8)
    return masks

def disk_kernel(radius_px, device, dtype=torch.float32, eps=1e-6):
    r = max(0.4, float(radius_px))
    R = int(math.ceil(r)) + 2
    y = torch.arange(-R, R+1, device=device, dtype=torch.int32)
    x = torch.arange(-R, R+1, device=device, dtype=torch.int32)
    y, x = torch.meshgrid(y, x, indexing='ij')
    k = ((x.to(torch.float32)**2 + y.to(torch.float32)**2) <= r*r + eps).to(dtype)
    return k / (k.sum() + 1e-8)

def chroma_radius(depth_m, ch_idx, fR=0.9, fG=1.0, fB=1.15, scale=120.0):
    f = [fR,fG,fB][ch_idx]
    return 1.0 + scale * abs(1.0/depth_m - 1.0/f)

class ChromaticPSFBank(nn.Module):
    def __init__(self, edges, fR=0.9, fG=1.0, fB=1.15, scale=120.0):
        super().__init__()
        self.register_buffer("edges", edges.float())
        self.J = self.edges.numel()-1
        self.fR, self.fG, self.fB, self.scale = fR, fG, fB, scale
        self._built = False

    def _build(self, device, dtype=torch.float32):
        self.kernels = []
        for j in range(self.J):
            zj = 2.0 / (1.0/self.edges[j] + 1.0/self.edges[j+1])  # harmonic mean depth
            per_ch = []
            for ch in range(3):
                rad = chroma_radius(float(zj), ch, self.fR, self.fG, self.fB, self.scale)
                per_ch.append(disk_kernel(rad, device, dtype=dtype))
            self.kernels.append(per_ch)
        self._built = True

    def forward(self, rgb, masks):
        rgb   = rgb.float()
        masks = masks.float()
        need_build = (
            (not self._built) or
            (self.kernels[0][0].device != rgb.device) or
            (self.kernels[0][0].dtype  != rgb.dtype)
        )
        if need_build:
            self._build(rgb.device, dtype=rgb.dtype)

        out = torch.zeros_like(rgb, dtype=rgb.dtype)
        for j in range(self.J):
            Mj = masks[:, j, 0:1].to(dtype=rgb.dtype)
            ch_imgs = []
            for ch in range(3):
                k = self.kernels[j][ch].to(device=rgb.device, dtype=rgb.dtype)[None, None]
                b = F.conv2d(rgb[:, ch:ch+1], k, padding=k.shape[-1]//2)
                ch_imgs.append(b)
            blurred = torch.cat(ch_imgs, dim=1)
            out += blurred * Mj
        return torch.clamp(out, 0.0, 1.0)

# ----------------------------
# U-Net (compact)
# ----------------------------

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, ch=[64,128,256,512,1024]):
        super().__init__()
        self.d1 = DoubleConv(3, ch[0]);  self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch[0], ch[1]);  self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch[1], ch[2]);  self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(ch[2], ch[3]);  self.p4 = nn.MaxPool2d(2)
        self.b  = DoubleConv(ch[3], ch[4])

        self.u4  = nn.ConvTranspose2d(ch[4], ch[3], 2, 2)
        self.u4d = DoubleConv(ch[4], ch[3])

        self.u3  = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.u3d = DoubleConv(ch[3], ch[2])

        self.u2  = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.u2d = DoubleConv(ch[2], ch[1])

        self.u1  = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.u1d = DoubleConv(ch[0] + ch[0], ch[0])

        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        b  = self.b(self.p4(d4))

        x = self.u4(b); x = self.u4d(torch.cat([x, d4], 1))
        x = self.u3(x); x = self.u3d(torch.cat([x, d3], 1))
        x = self.u2(x); x = self.u2d(torch.cat([x, d2], 1))
        x = self.u1(x); x = self.u1d(torch.cat([x, d1], 1))
        return self.out(x)  # log-depth

# ----------------------------
# Metrics
# ----------------------------

def depth_metrics(pred, gt):
    valid = gt > 0
    pred = pred[valid]; 
    gt = gt[valid]
    vmin, vmax = 0.3, 10.0
    pred = torch.clamp(pred, vmin, vmax); gt = torch.clamp(gt, vmin, vmax)
    abs_rel = torch.mean(torch.abs(pred-gt)/gt).item()
    rmse    = torch.sqrt(torch.mean((pred-gt)**2)).item()
    ratio = torch.max(pred/gt, gt/pred)
    d1 = (ratio < 1.25).float().mean().item()
    return dict(abs_rel=abs_rel, rmse=rmse, delta1=d1)

# ----------------------------
# Train / Infer
# ----------------------------

def save_checkpoint(path, net, edges, sim_params, img_size):
    ckpt = {
        "model": net.state_dict(),
        "edges": edges.detach().cpu(),
        "sim_params": sim_params,
        "img_size": img_size,
    }
    torch.save(ckpt, path)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    H, W = args.height, args.width
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = NYUNativeTrain(args.root, resize_hw=(H,W), center_crop=True, jitter=True)
    val_ds   = NYUNativeTest(args.root,  resize_hw=(H,W), center_crop=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              pin_memory=pin, persistent_workers=False)
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, persistent_workers=False)
    print(f"[info] train samples={len(train_ds)} val samples={len(val_ds)}")

    edges = make_invdepth_edges(0.5, 10.0, args.num_bins).to(device)
    simulator = ChromaticPSFBank(edges, fR=args.fR, fG=args.fG, fB=args.fB, scale=args.blur_scale).to(device)
    net = UNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    pbar = tqdm(range(args.iters), desc="train iters")
    it_train = iter(train_loader)

    _rgb, _depth, _ = next(it_train)
    print(f"[info] first batch ready: rgb {_rgb.shape}, depth {_depth.shape}")

    load_times, sim_times, net_times = [], [], []
    sim_params = dict(fR=args.fR, fG=args.fG, fB=args.fB, scale=args.blur_scale)

    for it in pbar:
        start_load = time.time()
        try:
            rgb, depth, _ = next(it_train)
        except StopIteration:
            it_train = iter(train_loader)
            rgb, depth, _ = next(it_train)
        load_times.append(time.time() - start_load)

        rgb = rgb.to(device).float()
        depth = depth.to(device).float().clamp(0.3, 10.0)

        start_sim = time.time()
        masks = depth_to_soft_masks(depth, edges)
        sim_rgb = simulator(rgb, masks)
        sim_times.append(time.time() - start_sim)

        start_net = time.time()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred_logd = net(sim_rgb)
                gt_logd = torch.log(depth.unsqueeze(1) + 1e-6)
                loss = F.mse_loss(pred_logd, gt_logd)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred_logd = net(sim_rgb)
            gt_logd = torch.log(depth.unsqueeze(1) + 1e-6)
            loss = F.mse_loss(pred_logd, gt_logd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net_times.append(time.time() - start_net)

        pbar.set_postfix(loss=float(loss.item()))

        if (it+1) % 10 == 0:
            avg_load = np.mean(load_times)
            avg_sim  = np.mean(sim_times)
            avg_net  = np.mean(net_times)
            est_total = (avg_load+avg_sim+avg_net) * args.iters / 60.0
            print(f"[timing] avg load={avg_load:.3f}s, sim={avg_sim:.3f}s, net={avg_net:.3f}s -> est total ≈ {est_total:.1f} min")

        if args.save_every and (it+1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"iter_{it+1}.pth")
            save_checkpoint(ckpt_path, net, edges, sim_params, (H,W))
            print(f"[ckpt] saved {ckpt_path}")

        if args.dry_run and (it+1) >= args.dry_run:
            print(f"[dry_run] Completed {args.dry_run} iterations. Exiting.")
            break

    ckpt_path = os.path.join(args.out_dir, "last.pth")
    save_checkpoint(ckpt_path, net, edges, sim_params, (H,W))
    print(f"[ckpt] saved final {ckpt_path}")

def infer_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    net = UNet().to(device); net.load_state_dict(ckpt["model"]); net.eval()
    edges = ckpt.get("edges", make_invdepth_edges()).to(device)
    sim_params = ckpt.get("sim_params", dict(fR=0.9,fG=1.0,fB=1.15,scale=120.0))
    simulator = ChromaticPSFBank(edges, **sim_params).to(device)

    H, W = ckpt.get("img_size", (args.height, args.width))
    test_ds = NYUNativeTest(args.root, resize_hw=(H,W), center_crop=True)

    pin = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=pin, persistent_workers=False
    )
    print(f"[info] test samples={len(test_ds)} workers={args.workers} pin_memory={pin}")

    # Select output dirs
    out_dir_fixed = args.out_dir
    out_dir_auto  = args.out_dir_auto if args.out_dir_auto else args.out_dir
    if args.viz_mode == "fixed":
        os.makedirs(out_dir_fixed, exist_ok=True)
    elif args.viz_mode == "auto":
        os.makedirs(out_dir_auto, exist_ok=True)

    with torch.no_grad():
        for rgb, depth, stem in tqdm(test_loader, desc="infer_test"):
            rgb    = rgb.to(device).float()
            ddepth = depth.to(device).float()
            masks  = depth_to_soft_masks(ddepth, edges)
            sim_rgb = simulator(rgb, masks)
            pred_logd = net(sim_rgb)
            pred_m = torch.exp(pred_logd).squeeze(0).squeeze(0).cpu().numpy()  # meters

            base = stem[0].replace("_colors","")

            # Save visualization(s)
            if args.viz_mode == "fixed":
                save_depth_png(pred_m, str(Path(out_dir_fixed)/f"{base}_pred_depth.png"),
                               mode="fixed", vmin=args.vmin, vmax=args.vmax)
            elif args.viz_mode == "auto":
                save_depth_png(pred_m, str(Path(out_dir_auto)/f"{base}_pred_depth.png"),
                               mode="auto", lo_p=args.viz_lo, hi_p=args.viz_hi)

            # Optional: meters as .npy
            if args.save_meters:
                np.save(str(Path(out_dir_auto if args.viz_mode=="auto" else out_dir_fixed)/f"{base}_pred_depth.npy"),
                        pred_m)

# ----------------------------
# CLI
# ----------------------------

def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--root", required=True, help="Folder that contains nyu2_train and nyu2_test")
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--iters", type=int, default=20000)
    tr.add_argument("--batch_size", type=int, default=3)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--height", type=int, default=240)
    tr.add_argument("--width", type=int, default=320)
    tr.add_argument("--num_bins", type=int, default=12)
    tr.add_argument("--fR", type=float, default=0.9)
    tr.add_argument("--fG", type=float, default=1.0)
    tr.add_argument("--fB", type=float, default=1.15)
    tr.add_argument("--blur_scale", type=float, default=120.0)
    tr.add_argument("--val_every", type=int, default=2000)
    tr.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    tr.add_argument("--dry_run", type=int, default=0, help="Run this many iterations then exit (debug)")
    tr.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N iters (0=off)")
    tr.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA AMP)")

    inf = sub.add_parser("infer_test")
    inf.add_argument("--root", required=True)
    inf.add_argument("--checkpoint", required=True)
    inf.add_argument("--out_dir", required=True, help="Output directory for PNGs")
    # viz controls
    inf.add_argument("--viz_mode", choices=["fixed","auto"], default="fixed",
                     help="fixed: clamp to [vmin,vmax]; auto: per-image percentile stretching")
    inf.add_argument("--vmin", type=float, default=0.3)
    inf.add_argument("--vmax", type=float, default=10.0)
    inf.add_argument("--viz_lo", type=float, default=1.0, help="auto-contrast low percentile")
    inf.add_argument("--viz_hi", type=float, default=99.0, help="auto-contrast high percentile")
    inf.add_argument("--out_dir_auto", type=str, default="", help="Optional separate dir for auto viz")
    inf.add_argument("--save_meters", action="store_true", help="Also save .npy depth in meters")
    # shape & perf
    inf.add_argument("--height", type=int, default=240)
    inf.add_argument("--width", type=int, default=320)
    inf.add_argument("--workers", type=int, default=0)

    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer_test":
        infer_test(args)
