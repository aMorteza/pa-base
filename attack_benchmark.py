#!/usr/bin/env python3
"""
Empirical attack benchmark for PA-BSE  (action item: attack study)

Methodology follows P3SL [Fan et al., IEEE TNSE 2026], which evaluates a split
system by running a data-reconstruction attack against the transmitted
representation and scoring the recovered image against the original. P3SL uses
the UnSplit attack scored by FSIM; we use the standard learned-inversion attack
(Dosovitskiy & Brox; Mahendran & Vedaldi) which is the appropriate analogue for
*inference* rather than training, and score with SSIM, PSNR and MSE. Higher
similarity to the original input means more leakage.

The point of the study is comparative, exactly as in P3SL: we run the SAME
attack against
  (a) the privacy-UNAWARE system  (sigma = 0), and
  (b) the privacy-AWARE system    (sigma > 0, at several per-element budgets),
so the reported difference is what privacy awareness actually buys. We also
sweep the split layer at fixed noise, to separate the effect of depth from the
effect of the mechanism.

Threat model (matches the paper): an honest-but-curious server observes the
released representation Y_l and knows the architecture, the split layer, the
mechanism and sigma. It may train an inversion network on its own data.
This is a STRONGER attacker than UnSplit, which is data-free, so the numbers
here are conservative.

Run:
    export MPLBACKEND=Agg
    export CUDA_VISIBLE_DEVICES=1
    python attack_benchmark.py 2>&1 | tee attack_run.log
"""
import os, json, math, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm, torchvision.datasets as tvd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.abspath(os.environ.get("PA_BSE_RESULTS", "results"))
OUT = os.path.join(RESULTS_DIR, "attack"); os.makedirs(OUT, exist_ok=True)
print(f"[attack] device={DEVICE}  out={OUT}")

# ----------------------------------------------------------------- config
class ACFG:
    c0            = 1.0        # per-coordinate clip budget (matches the optimizer)
    train_images  = 4000       # attacker training set size
    test_images   = 500        # held-out images for reporting
    batch         = 32
    epochs        = 12         # inversion-decoder training epochs
    lr            = 1e-3
    # split layers to study: the PA-BSE optimum, the non-private optimum,
    # a shallow and a deeper feasible layer
    layers        = [0, 4, 6, 13]
    # per-element budgets to study; sigma is the smallest value meeting each
    eps_bars      = [0.5, 0.75, 1.0, 2.0]
    seed          = 42

def sigma_for(eps_bar, c0=ACFG.c0):
    """Smallest sigma meeting the per-element budget: sigma >= c0/sqrt(2^(2 eps)-1)."""
    return c0/math.sqrt(2**(2*eps_bar) - 1)

VGG19_NAMES = ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
 "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
 "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","conv3_4","relu3_4","pool3",
 "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","conv4_4","relu4_4","pool4",
 "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","conv5_4","relu5_4","pool5"]

# ----------------------------------------------------------------- model + data
weights = tvm.VGG19_Weights.DEFAULT
model = tvm.vgg19(weights=weights).to(DEVICE).eval()
for p in model.parameters(): p.requires_grad_(False)
transform = weights.transforms()

def load_split(split, n):
    import kagglehub
    root = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    ds = tvd.ImageFolder(os.path.join(root, "imagenet-mini", split), transform=transform)
    g = np.random.default_rng(ACFG.seed)
    idx = g.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs = [ds[int(i)][0] for i in idx]
    return torch.stack(xs)

print("[attack] loading data ...")
X_train = load_split("train", ACFG.train_images)
X_test  = load_split("val",   ACFG.test_images)
print(f"[attack] attacker train {tuple(X_train.shape)}  test {tuple(X_test.shape)}")

@torch.no_grad()
def forward_head(x, l):
    for m in model.features[:l+1]: x = m(x)
    return x

def privatize(z, sigma, c0=ACFG.c0):
    """clip_{C_l}(z) + N(0, sigma^2 I), C_l = c0*sqrt(d_l). sigma=0 -> identity,
    exactly as in the optimizer, so the sigma=0 case is the privacy-unaware system."""
    if sigma <= 0: return z
    f = z.flatten(1); d = f.shape[1]; C = c0*(d**0.5)
    n = f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    f = f*torch.clamp(C/n, max=1.0)
    return (f + torch.randn_like(f)*sigma).view_as(z)

# ----------------------------------------------------------------- attacker
class Inverter(nn.Module):
    """Convolutional decoder mapping the released representation back to the input.
    Upsamples to 224x224 regardless of the split layer's spatial size."""
    def __init__(self, in_ch, in_hw):
        super().__init__()
        ups = max(1, int(round(math.log2(224/in_hw)))) if in_hw < 224 else 0
        chans, layers, c = [256,128,64,32,16], [], in_ch
        layers += [nn.Conv2d(in_ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)]
        c = 256
        for i in range(ups):
            nc = chans[min(i+1, len(chans)-1)]
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(c, nc, 3, padding=1), nn.BatchNorm2d(nc), nn.ReLU(True)]
            c = nc
        layers += [nn.Conv2d(c, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        y = self.net(z)
        if y.shape[-1] != 224:
            y = F.interpolate(y, size=(224,224), mode="bilinear", align_corners=False)
        return y

def ssim(a, b, C1=0.01**2, C2=0.03**2):
    """Mean SSIM over a batch, computed on the mean channel with an 11x11 uniform window."""
    a = a.mean(1, keepdim=True); b = b.mean(1, keepdim=True)
    w = torch.ones(1,1,11,11, device=a.device)/121.0
    mu_a, mu_b = F.conv2d(a,w,padding=5), F.conv2d(b,w,padding=5)
    va = F.conv2d(a*a,w,padding=5)-mu_a**2; vb = F.conv2d(b*b,w,padding=5)-mu_b**2
    vab = F.conv2d(a*b,w,padding=5)-mu_a*mu_b
    s = ((2*mu_a*mu_b+C1)*(2*vab+C2))/((mu_a**2+mu_b**2+C1)*(va+vb+C2))
    return s.mean().item()

def train_and_eval(l, sigma, tag):
    """Train the inversion attacker on (representation -> input) and score it."""
    torch.manual_seed(ACFG.seed)
    with torch.no_grad():
        probe = forward_head(X_train[:2].to(DEVICE), l)
    in_ch, in_hw = probe.shape[1], probe.shape[-1]
    net = Inverter(in_ch, in_hw).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=ACFG.lr)
    n = X_train.shape[0]
    for ep in range(ACFG.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, ACFG.batch):
            xb = X_train[perm[i:i+ACFG.batch]].to(DEVICE)
            with torch.no_grad():
                z = privatize(forward_head(xb, l), sigma)   # what the server sees
            rec = net(z)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.shape[0]
        if ep == ACFG.epochs-1:
            print(f"    [{tag}] final train MSE {tot/n:.4f}")
    # ---- evaluate on held-out images ----
    net.eval(); mses, ssims, psnrs = [], [], []
    with torch.no_grad():
        for i in range(0, X_test.shape[0], ACFG.batch):
            xb = X_test[i:i+ACFG.batch].to(DEVICE)
            rec = net(privatize(forward_head(xb, l), sigma))
            m = F.mse_loss(rec, xb).item()
            mses.append(m); ssims.append(ssim(rec, xb))
            psnrs.append(10*math.log10(1.0/max(m,1e-12)))
    return dict(mse=float(np.mean(mses)), ssim=float(np.mean(ssims)),
                psnr=float(np.mean(psnrs))), net

# ================================================================= STUDY 1
# Privacy-unaware vs privacy-aware at the PA-BSE operating point and others.
print("\n" + "="*70); print("STUDY 1 - attack success vs privacy budget"); print("="*70)
rows = []
L_STAR = 6                                    # PA-BSE optimum split layer
for eb in [None] + ACFG.eps_bars:             # None = privacy-unaware (sigma=0)
    s = 0.0 if eb is None else sigma_for(eb)
    tag = "no privacy (sigma=0)" if eb is None else f"eps_bar={eb} (sigma={s:.3f})"
    print(f"  training attacker: layer {L_STAR}, {tag}")
    m, _ = train_and_eval(L_STAR, s, tag)
    rows.append(dict(setting=tag, layer=L_STAR, eps_bar=("none" if eb is None else eb),
                     sigma=round(s,4), **{k: round(v,4) for k,v in m.items()}))
    print(f"    -> SSIM {m['ssim']:.4f}  PSNR {m['psnr']:.2f} dB  MSE {m['mse']:.4f}")
S1 = pd.DataFrame(rows); S1.to_csv(f"{OUT}/attack_vs_budget.csv", index=False)
base = S1[S1.eps_bar=="none"].iloc[0]
S1["ssim_drop_vs_unaware"] = (base.ssim - S1.ssim).round(4)
S1.to_csv(f"{OUT}/attack_vs_budget.csv", index=False)
print("\n"+S1.to_string(index=False))

# ================================================================= STUDY 2
# Effect of split depth at fixed budget, and with no privacy.
print("\n" + "="*70); print("STUDY 2 - attack success vs split layer"); print("="*70)
rows2 = []
s_star = sigma_for(1.0)
for l in ACFG.layers:
    for s, lab in [(0.0, "no privacy"), (s_star, f"eps_bar=1.0")]:
        print(f"  training attacker: layer {l} ({VGG19_NAMES[l]}), {lab}")
        m, _ = train_and_eval(l, s, f"l{l}-{lab}")
        rows2.append(dict(layer=l, name=VGG19_NAMES[l], setting=lab, sigma=round(s,4),
                          **{k: round(v,4) for k,v in m.items()}))
        print(f"    -> SSIM {m['ssim']:.4f}  PSNR {m['psnr']:.2f} dB")
S2 = pd.DataFrame(rows2); S2.to_csv(f"{OUT}/attack_vs_layer.csv", index=False)
print("\n"+S2.to_string(index=False))

# ================================================================= FIGURES
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
fin = S1[S1.eps_bar!="none"].copy(); fin["eps_bar"] = fin["eps_bar"].astype(float)
ax[0].axhline(base.ssim, ls="--", color="tab:red",
              label=f"privacy-unaware ({base.ssim:.3f})")
ax[0].plot(fin.eps_bar, fin.ssim, "o-", color="tab:blue", label="PA-BSE mechanism")
ax[0].set_xlabel("per-element budget $\\bar\\epsilon$ (bits/element)")
ax[0].set_ylabel("reconstruction SSIM (higher = more leakage)")
ax[0].set_title(f"Attack success vs privacy budget (split layer {L_STAR})")
ax[0].legend(); ax[0].grid(alpha=.3)
w = 0.38; xs = np.arange(len(ACFG.layers))
u = [S2[(S2.layer==l)&(S2.setting=="no privacy")].ssim.iloc[0] for l in ACFG.layers]
p = [S2[(S2.layer==l)&(S2.setting!="no privacy")].ssim.iloc[0] for l in ACFG.layers]
ax[1].bar(xs-w/2, u, w, label="privacy-unaware", color="tab:red")
ax[1].bar(xs+w/2, p, w, label="PA-BSE ($\\bar\\epsilon$=1.0)", color="tab:blue")
ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{l}\n{VGG19_NAMES[l]}" for l in ACFG.layers], fontsize=8)
ax[1].set_xlabel("split layer"); ax[1].set_ylabel("reconstruction SSIM")
ax[1].set_title("Attack success vs split layer"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/attack_benchmark.png", dpi=140, bbox_inches="tight")
plt.savefig(f"{OUT}/attack_benchmark.pdf", bbox_inches="tight"); plt.close()

# qualitative panel: originals vs reconstructions, unaware vs aware
print("\n[attack] rendering qualitative reconstructions ...")
_, net_u = train_and_eval(L_STAR, 0.0, "qual-unaware")
_, net_p = train_and_eval(L_STAR, s_star, "qual-aware")
with torch.no_grad():
    xb = X_test[:6].to(DEVICE)
    ru = net_u(privatize(forward_head(xb, L_STAR), 0.0)).clamp(0,1).cpu()
    rp = net_p(privatize(forward_head(xb, L_STAR), s_star)).clamp(0,1).cpu()
    xo = xb.clamp(0,1).cpu()
fig, axes = plt.subplots(3, 6, figsize=(13, 6.8))
for j in range(6):
    for i, (img, lab) in enumerate([(xo[j],"original"),(ru[j],"attack, no privacy"),
                                    (rp[j],"attack, PA-BSE")]):
        axes[i,j].imshow(img.permute(1,2,0).numpy()); axes[i,j].axis("off")
        if j==0: axes[i,j].set_title(lab, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/attack_qualitative.png", dpi=140, bbox_inches="tight")
plt.savefig(f"{OUT}/attack_qualitative.pdf", bbox_inches="tight"); plt.close()

# ================================================================= SUMMARY
print("\n" + "="*70); print("SUMMARY"); print("="*70)
best_priv = fin.loc[fin.ssim.idxmin()]
print(f"privacy-unaware reconstruction SSIM : {base.ssim:.4f}")
print(f"PA-BSE at eps_bar=1.0               : "
      f"{float(S1[S1.eps_bar==1.0].ssim.iloc[0]):.4f}")
print(f"largest reduction (eps_bar={best_priv.eps_bar}) : {best_priv.ssim:.4f} "
      f"({base.ssim-best_priv.ssim:+.4f} vs unaware)")
print("\nInterpretation: lower SSIM means the attacker recovers less of the input.")
print("The analytical per-element bound is an upper bound on leakage; these")
print("measurements show what an actual attacker achieves at each budget.")
json.dump(dict(study1=S1.to_dict("records"), study2=S2.to_dict("records"),
               unaware_ssim=float(base.ssim)),
          open(f"{OUT}/attack_summary.json","w"), indent=2, default=float)
print(f"\nSaved to {OUT}")
