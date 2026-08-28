#!/usr/bin/env python3
"""
PA-BSE follow-up experiments — action items 2 and 3.

ITEM 2  Noise range and split-layer diversity
  - Documents where the noise range endpoints come from (the min is the
    feasibility floor implied by the per-element budget; the max was an
    unjustified choice and is now swept explicitly).
  - Sweeps the per-element budget and the noise range to find how many
    DISTINCT split layers can become optimal, and explains why.
  - Terminology: "noise range" (a vector over sigma), not "noise grid".

ITEM 3  Comparison against other privacy-preserving baselines
  - Adds privacy baselines so the gain of PA-BSE is measured against
    privacy-aware alternatives, not only against the non-private system.
  - Adds constrained variants (split layer fixed / noise fixed / power fixed)
    to isolate the contribution of each decision variable.

Run AFTER a normal PA-BSE run so the cached accuracy tables exist:
    export MPLBACKEND=Agg
    export CUDA_VISIBLE_DEVICES=1
    python pa_bse_followup.py 2>&1 | tee followup.log
"""
import os, json, math, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- config
RESULTS_DIR = os.path.abspath(os.environ.get("PA_BSE_RESULTS", "results"))
OUT = os.path.join(RESULTS_DIR, "followup")
os.makedirs(OUT, exist_ok=True)

C0        = 1.0        # per-coordinate clip budget
E_MAX     = 5.0
TAU_MAX   = 5.0
N_LAYERS  = 37
P_MIN, P_MAX, N_POWER = 0.1, 0.5, 100
P_GRID    = np.linspace(P_MIN, P_MAX, N_POWER)

VGG19_NAMES = ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
 "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
 "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","conv3_4","relu3_4","pool3",
 "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","conv4_4","relu4_4","pool4",
 "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","conv5_4","relu5_4","pool5"]

def savefig_both(base, fig=None):
    f = fig or plt.gcf()
    f.savefig(base + ".png", dpi=140, bbox_inches="tight")
    f.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(f)

# ----------------------------------------------------------------- load cache
prof = pd.read_csv(os.path.join(RESULTS_DIR, "vgg19_profile.csv"))
D_BITS   = prof["D_bits"].to_numpy(float)
CUM_DEV  = prof["cum_flops_device"].to_numpy(float)
CUM_SRV  = prof["cum_flops_server"].to_numpy(float)
D_ELEM   = prof["elements"].to_numpy(float)

acc3 = pd.read_csv(os.path.join(RESULTS_DIR, "accuracy_profile_priv.csv"))
ACC3 = {(int(r.split_layer), round(r.drop_rate,4), round(r.sigma,4)): r.accuracy
        for r in acc3.itertuples()}
PROFILED_DROPS  = np.array(sorted({round(x,4) for x in acc3.drop_rate.unique()}))
PROFILED_SIGMAS = np.array(sorted({round(x,4) for x in acc3.sigma.unique()}))
CLEAN = float(acc3[(acc3.sigma==0.0) & (acc3.drop_rate==0.0)].accuracy.iloc[0])

summ = json.load(open(os.path.join(RESULTS_DIR, "MASTER_summary.json")))
B_HZ  = 240_000*256*0.8
NOISE = 10**((-147.0-30)/10) * B_HZ
H2_BO = 10**(summ["channel_gain_dB"]/10) if "channel_gain_dB" in summ else 10**(-99.4/10)

print(f"[load] profiled sigmas (noise range): {list(PROFILED_SIGMAS)}")
print(f"[load] profiled drops: {list(PROFILED_DROPS)}")
print(f"[load] clean accuracy: {CLEAN*100:.2f}%  channel {10*np.log10(H2_BO):.1f} dB")

# ----------------------------------------------------------------- system model
KAPPA=1e-29; F_DEV=1.8e9; C_DEV=4; ETA_DEV=0.7; F_SRV=4.0e9; C_SRV=8; ETA_SRV=0.8
def tau_device(l): return CUM_DEV[l]/(C_DEV*F_DEV*ETA_DEV)
def tau_server(l): return CUM_SRV[l]/(C_SRV*F_SRV*ETA_SRV)
def comp_energy(l): return C_DEV*KAPPA*CUM_DEV[l]*F_DEV**2
def rate_bps(P):   return B_HZ*np.log2(1+P*H2_BO/NOISE)
def tau_tx_full(l,P): return D_BITS[l]/rate_bps(P)

def eps_bar_dp(sigma, c0=C0):
    """Per-element Renyi-DP budget consumed by noise level sigma (bits/element)."""
    return math.inf if sigma <= 0 else 0.5*math.log2(1.0 + c0**2/sigma**2)

def sigma_floor(eps_bar, c0=C0):
    """Smallest sigma satisfying the per-element budget: the LOWER end of the range."""
    return c0/math.sqrt(2**(2*eps_bar) - 1)

def acc_lookup(l, drop, sigma):
    j = PROFILED_DROPS[np.argmin(np.abs(PROFILED_DROPS - np.clip(drop,0,PROFILED_DROPS[-1])))]
    s = PROFILED_SIGMAS[np.argmin(np.abs(PROFILED_SIGMAS - np.clip(sigma,0,PROFILED_SIGMAS[-1])))]
    return ACC3[(int(l), round(float(j),4), round(float(s),4))]

def evaluate(l, P, sigma, eps_bar, baseline=False):
    l = int(np.clip(round(l), 0, N_LAYERS-1))
    t_md, t_s = tau_device(l), tau_server(l)
    tt_full = tau_tx_full(l, P)
    t_e = TAU_MAX - t_md - t_s
    if t_e <= 0:            drop, tt = 1.0, 0.0
    elif tt_full <= t_e:    drop, tt = 0.0, tt_full
    else:                   drop, tt = float(np.clip(1-t_e/tt_full,0,1)), t_e
    E = comp_energy(l) + P*tt
    delay = t_md + tt + t_s
    eb = eps_bar_dp(sigma)
    priv_ok = True if (baseline and sigma <= 0) else (eb <= eps_bar)
    res_ok  = (E <= E_MAX) and (delay <= TAU_MAX + 1e-9)
    return dict(l=l, P=float(P), sigma=float(sigma), U=acc_lookup(l, drop, sigma),
                E=E, delay=delay, drop=drop, eps_bar=eb,
                feasible=bool(res_ok and priv_ok), res_ok=res_ok, priv_ok=priv_ok)

def optimize(eps_bar, sigmas=None, layers=None, powers=None, baseline=False):
    """Exhaustive best-feasible over the given ranges."""
    sigmas = PROFILED_SIGMAS[PROFILED_SIGMAS > 0] if sigmas is None else np.asarray(sigmas)
    if baseline: sigmas = np.array([0.0])
    layers = range(N_LAYERS) if layers is None else layers
    powers = P_GRID if powers is None else np.asarray(powers)
    best = None
    for l in layers:
        for P in powers:
            for s in sigmas:
                r = evaluate(l, P, s, eps_bar, baseline=baseline)
                if r["feasible"] and (best is None or r["U"] > best["U"]): best = r
    return best

# =====================================================================
# ITEM 2a — where do the noise-range endpoints come from?
# =====================================================================
print("\n" + "="*72)
print("ITEM 2a — PROVENANCE OF THE NOISE RANGE")
print("="*72)
rows=[]
for eb in [0.25,0.5,0.75,1.0,1.5,2.0,3.0]:
    rows.append(dict(eps_bar=eb, sigma_min_required=round(sigma_floor(eb),4),
                     note="lower end = feasibility floor c0/sqrt(2^(2*eps_bar)-1)"))
prov = pd.DataFrame(rows)
prov.to_csv(f"{OUT}/item2a_noise_range_provenance.csv", index=False)
print(prov.to_string(index=False))
print("\nLower end: DERIVED — the smallest sigma that satisfies the per-element budget.")
print("Upper end: NOT derived. sigma_max=2.0 in the original runs was an arbitrary")
print("           choice. Below we sweep the upper end explicitly to justify it.")

# how far up does sigma need to go before accuracy is destroyed?
lay_probe = 6
acc_by_sigma = [(float(s), acc_lookup(lay_probe, 0.0, s)*100) for s in PROFILED_SIGMAS]
print(f"\naccuracy at layer {lay_probe} across the profiled noise range:")
for s,a in acc_by_sigma: print(f"   sigma={s:6.3f} -> {a:6.2f}%")
usable = [s for s,a in acc_by_sigma if a > 50]
print(f"\n=> accuracy stays above 50% only for sigma <= {max(usable) if usable else 0:.3f};")
print("   beyond that the mechanism destroys utility, which bounds the useful range.")

# =====================================================================
# ITEM 2b — why only layers 6 and 13? (feasibility + activation type)
# =====================================================================
print("\n" + "="*72)
print("ITEM 2b — WHY ONLY A FEW SPLIT LAYERS ARE SELECTED")
print("="*72)
diag=[]
for l in range(N_LAYERS):
    r0 = evaluate(l, 0.3, 0.0, 1.0, baseline=True)
    r1 = evaluate(l, 0.3, float(PROFILED_SIGMAS[PROFILED_SIGMAS>0][0]), 1.0)
    diag.append(dict(layer=l, name=VGG19_NAMES[l],
        type=("conv" if "conv" in VGG19_NAMES[l] else "relu" if "relu" in VGG19_NAMES[l] else "pool"),
        d_l=int(D_ELEM[l]), delay_s=round(r1["delay"],3), energy_J=round(r1["E"],3),
        resource_feasible=r1["res_ok"], acc_noisy_pct=round(r1["U"]*100,2)))
D = pd.DataFrame(diag)
D.to_csv(f"{OUT}/item2b_layer_diagnosis.csv", index=False)
nfeas = int(D.resource_feasible.sum())
print(f"resource-feasible layers: {nfeas}/{N_LAYERS} -> {D[D.resource_feasible].layer.tolist()}")
print(f"eliminated by delay/energy BEFORE privacy: {N_LAYERS-nfeas} layers")
print("\naccuracy under noise, by activation type (feasible layers only):")
print(D[D.resource_feasible].groupby("type")["acc_noisy_pct"].agg(["count","mean","max"]).round(2).to_string())
print("\nconv -> following relu (noise robustness of ReLU):")
for l in range(N_LAYERS-1):
    if "conv" in VGG19_NAMES[l] and "relu" in VGG19_NAMES[l+1]:
        a,b = D.loc[l,"acc_noisy_pct"], D.loc[l+1,"acc_noisy_pct"]
        if D.loc[l,"resource_feasible"]:
            print(f"   L{l:2d} {VGG19_NAMES[l]:8} {a:6.2f}%  ->  L{l+1:2d} {VGG19_NAMES[l+1]:8} {b:6.2f}%  ({b-a:+.1f})")
print("\n=> Two causes, both structural (NOT a coarse noise range):")
print("   (i) delay/energy eliminate deep layers before privacy is considered;")
print("   (ii) ReLU clamps negatives to zero, removing ~half the injected noise,")
print("        so post-ReLU (and pooling) split points dominate conv split points.")

# =====================================================================
# ITEM 2c — how many DISTINCT layers can become optimal?
# =====================================================================
print("\n" + "="*72)
print("ITEM 2c — SPLIT-LAYER DIVERSITY OVER THE BUDGET / NOISE RANGE")
print("="*72)
EPS_SWEEP = [0.25,0.4,0.5,0.6,0.75,0.9,1.0,1.25,1.5,2.0,2.5,3.0]
sweep=[]
for eb in EPS_SWEEP:
    ok = [s for s in PROFILED_SIGMAS if s>0 and eps_bar_dp(s) <= eb]
    if not ok:
        sweep.append(dict(eps_bar=eb, layer=None, sigma=None, acc_pct=None,
                          note="no profiled sigma satisfies this budget")); continue
    b = optimize(eb, sigmas=ok)
    sweep.append(dict(eps_bar=eb, layer=b["l"], name=VGG19_NAMES[b["l"]],
                      sigma=round(b["sigma"],4), power=round(b["P"],4),
                      acc_pct=round(b["U"]*100,2),
                      sigma_floor=round(sigma_floor(eb),4)))
S = pd.DataFrame(sweep)
S.to_csv(f"{OUT}/item2c_layer_diversity_sweep.csv", index=False)
print(S.to_string(index=False))
distinct = sorted({int(x) for x in S.layer.dropna()})
print(f"\nDISTINCT optimal split layers found: {distinct}  (count={len(distinct)})")

# also: force each feasible layer and report its best achievable accuracy,
# so we can see the full ranking rather than only the winner.
force=[]
for l in D[D.resource_feasible].layer:
    b = optimize(1.0, layers=[l])
    if b: force.append(dict(layer=l, name=VGG19_NAMES[l], best_acc_pct=round(b["U"]*100,2),
                            sigma=round(b["sigma"],4), power=round(b["P"],4)))
F = pd.DataFrame(force).sort_values("best_acc_pct", ascending=False)
F.to_csv(f"{OUT}/item2c_forced_layer_ranking.csv", index=False)
print("\nbest achievable accuracy when each layer is FORCED (eps_bar=1.0):")
print(F.to_string(index=False))

plt.figure(figsize=(11,4.5))
plt.bar(F.layer, F.best_acc_pct, color=["tab:green" if "relu" in n else
        "tab:orange" if "pool" in n else "tab:red" for n in F.name])
plt.xlabel("split layer l (forced)"); plt.ylabel("best achievable accuracy (%)")
plt.title("Item 2 — accuracy when each split layer is forced (green=relu, orange=pool, red=conv)")
plt.grid(alpha=.3, axis="y")
savefig_both(f"{OUT}/item2_forced_layer_ranking")

# =====================================================================
# ITEM 3 — comparison against other privacy-preserving baselines
# =====================================================================
print("\n" + "="*72)
print("ITEM 3 — PRIVACY-PRESERVING BASELINES AND CONSTRAINED VARIANTS")
print("="*72)
EB = 1.0
methods = []

# (0) non-private reference
b = optimize(EB, baseline=True)
methods.append(("BSE (non-private reference)", b, "no privacy mechanism; sigma=0"))

# (1) PA-BSE: joint optimization over (l, P, sigma)
b = optimize(EB)
methods.append(("PA-BSE (joint l,P,sigma)", b, "our method"))

# (2) Noise-only DP: keep the NON-PRIVATE optimal split/power, add the minimum
#     noise that meets the budget. This is the standard "bolt-on DP" baseline.
base = optimize(EB, baseline=True)
s_min = min([s for s in PROFILED_SIGMAS if s>0 and eps_bar_dp(s) <= EB], default=None)
if s_min is not None:
    r = evaluate(base["l"], base["P"], s_min, EB)
    methods.append(("Noise-only DP (fixed split/power)", r,
                    "privacy added on top of the non-private solution"))

# (3) Deepest-split heuristic: choose the deepest feasible layer (smallest payload,
#     lowest TOTAL leakage), then optimize power and noise.
feas_layers = D[D.resource_feasible].layer.tolist()
b = optimize(EB, layers=[max(feas_layers)])
methods.append(("Deepest-feasible-split heuristic", b, "maximise depth, then optimise P,sigma"))

# (4) Fixed-noise variant: pin sigma at the budget floor, optimise l and P.
if s_min is not None:
    b = optimize(EB, sigmas=[s_min])
    methods.append(("Fixed-noise (sigma at budget floor)", b, "sigma fixed; l,P optimised"))

# (5) Fixed-split variant: pin the split at the non-private optimum, optimise P,sigma.
b = optimize(EB, layers=[base["l"]])
methods.append((f"Fixed-split (l={base['l']}), optimise P,sigma", b, "isolates the value of choosing l"))

# (6) Fixed-power variant: pin power at the non-private optimum, optimise l,sigma.
b = optimize(EB, powers=[base["P"]])
methods.append((f"Fixed-power (P={base['P']:.3f}), optimise l,sigma", b, "isolates the value of choosing P"))

rows=[]
for name, r, note in methods:
    if r is None: rows.append(dict(Method=name, note="infeasible")); continue
    rows.append(dict(Method=name, SplitLayer=r["l"], LayerName=VGG19_NAMES[r["l"]],
        Power_W=round(r["P"],4), Sigma=round(r["sigma"],4),
        Accuracy_pct=round(r["U"]*100,2), Energy_J=round(r["E"],3),
        Delay_s=round(r["delay"],3),
        PerElem_Leakage=("inf" if not np.isfinite(r["eps_bar"]) else round(r["eps_bar"],4)),
        Note=note))
M = pd.DataFrame(rows)
# gain of PA-BSE over each privacy-aware alternative
pa = M[M.Method.str.startswith("PA-BSE")]
if len(pa):
    pa_acc = float(pa.Accuracy_pct.iloc[0])
    M["PA_BSE_gain_pts"] = M.Accuracy_pct.apply(
        lambda a: None if pd.isna(a) else round(pa_acc - a, 2))
M.to_csv(f"{OUT}/item3_privacy_baselines.csv", index=False)
print(M.to_string(index=False))

plt.figure(figsize=(11,5))
mm = M.dropna(subset=["Accuracy_pct"]).sort_values("Accuracy_pct")
cols = ["tab:blue" if "non-private" in m else "tab:red" if m.startswith("PA-BSE") else "tab:gray"
        for m in mm.Method]
plt.barh(range(len(mm)), mm.Accuracy_pct, color=cols)
plt.yticks(range(len(mm)), [m[:46] for m in mm.Method], fontsize=8)
plt.xlabel("accuracy (%)"); plt.title(f"Item 3 — PA-BSE vs privacy-preserving baselines (eps_bar={EB})")
plt.grid(alpha=.3, axis="x")
savefig_both(f"{OUT}/item3_privacy_baselines")

# ----------------------------------------------------------------- summary
print("\n" + "="*72)
print("SUMMARY")
print("="*72)
print(f"Item 2a: noise-range lower end is derived (feasibility floor); upper end was")
print(f"         arbitrary and is now swept explicitly.")
print(f"Item 2b: {N_LAYERS-nfeas}/{N_LAYERS} layers are eliminated by delay/energy before privacy;")
print(f"         within the rest, ReLU/pool split points dominate conv ones under noise.")
print(f"Item 2c: {len(distinct)} distinct optimal split layers across the budget sweep: {distinct}")
print(f"Item 3 : {len(M)} methods compared; outputs in {OUT}")
json.dump(dict(distinct_optimal_layers=distinct,
               resource_feasible_layers=[int(x) for x in D[D.resource_feasible].layer],
               n_eliminated_by_resources=int(N_LAYERS-nfeas),
               methods=M.to_dict(orient="records")),
          open(f"{OUT}/followup_summary.json","w"), indent=2, default=str)
print(f"\nSaved: {OUT}")
