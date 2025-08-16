#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RD9–weighted entropy method helper:
- H2, Δ(p) = H2(2p - p^2) - H2(p)
- p_max (argmax of Δ on [0, 1/2])
- m_window(γ, β) = min_{p in [γ, β]} Δ(p)
- Solve θ*m_window + (1-θ)*Δ(α) = 0  for α_max in [α*, 1/2)
- (Optional) sweep over β and save CSV/plot

Usage examples:
  python rd9_entropy_window.py --theta 0.33 --gamma 0.16 --beta 0.34
  python rd9_entropy_window.py --theta 0.44 --gamma 0.16 --beta 0.34 --grid
"""

import math
import argparse
from typing import Tuple, Optional, List

try:
    import numpy as np
except ImportError:
    raise SystemExit("Este script requiere numpy instalado.")

# Matplotlib es opcional (solo para figuras)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

ALPHA_STAR = (3 - math.sqrt(5)) / 2  # ≈ 0.38196601125

def H2(p: float) -> float:
    """Binary entropy in bits, with safe endpoints."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p*math.log(p, 2) - (1-p)*math.log(1-p, 2)

def T(p: float) -> float:
    """Union map for a single coordinate: p -> 2p - p^2."""
    return 2*p - p*p

def Delta(p: float) -> float:
    """Δ(p) = H2(T(p)) - H2(p)."""
    return H2(T(p)) - H2(p)

def argmax_delta_on_unit_half(n=20001) -> Tuple[float, float]:
    """Brute-force argmax of Δ on [0, 1/2]."""
    grid = np.linspace(0.0, 0.5, n)
    vals = np.array([Delta(x) for x in grid])
    k = int(np.argmax(vals))
    return float(grid[k]), float(vals[k])

def m_window(gamma: float, beta: float, n=20001) -> float:
    """
    m(γ, β) = min_{p in [γ, β]} Δ(p), assuming 0 ≤ γ ≤ β ≤ 1/2.
    If γ < 0, clamped to 0. If β > 1/2, clamped to 1/2.
    """
    lo = max(0.0, gamma)
    hi = min(0.5, beta)
    if hi < lo:
        raise ValueError("Se requiere 0 ≤ γ ≤ β ≤ 1/2.")
    grid = np.linspace(lo, hi, n)
    vals = np.array([Delta(x) for x in grid])
    return float(np.min(vals))

def find_root_bisection(f, a: float, b: float, tol=1e-12, maxit=200) -> Optional[float]:
    """Simple bisection finder. Requires f(a)*f(b) ≤ 0."""
    fa, fb = f(a), f(b)
    if math.isnan(fa) or math.isnan(fb):
        return None
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0:
        return None
    lo, hi = a, b
    for _ in range(maxit):
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if fm == 0.0 or (hi - lo) < tol:
            return mid
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return 0.5*(lo+hi)

def alpha_max(theta: float, gamma: float, beta: float) -> float:
    """
    Solve θ*m(γ,β) + (1-θ)*Δ(α) = 0 for α ∈ [α*, 1/2).
    If no sign change, returns α*.
    """
    m = m_window(gamma, beta)
    def F(a):
        return theta*m + (1.0 - theta)*Delta(a)
    # Search on [ALPHA_STAR, 0.499999]
    a_lo, a_hi = ALPHA_STAR, 0.499999
    root = find_root_bisection(F, a_lo, a_hi)
    if root is None:
        return ALPHA_STAR
    return float(root)

def sweep_beta(theta: float, gamma: float, betas: np.ndarray) -> List[Tuple[float, float]]:
    out = []
    for b in betas:
        amax = alpha_max(theta, gamma, float(b))
        out.append((float(b), amax))
    return out

def save_csv(path: str, rows: List[Tuple[float, float]], header=("beta", "alpha_max")):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def plot_delta_shape(pmax: float, dmax: float, path_png="Delta_shape.png"):
    if not HAS_MPL:
        return
    ps = np.linspace(0, 0.5, 2001)
    vals = [Delta(p) for p in ps]
    plt.figure(figsize=(7,4))
    plt.plot(ps, vals, label="Δ(p)")
    plt.axvline(pmax, linestyle="--", label=f"$p_\\max \\approx {pmax:.3f}$")
    plt.scatter([pmax],[dmax], zorder=5)
    plt.xlabel("p")
    plt.ylabel("Δ(p)")
    plt.title(r"Shape of $\Delta(p)=H_2(2p-p^2)-H_2(p)$ on $[0,1/2]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def plot_alpha_vs_beta(curves, theta_list, gamma_list, betas, path_png="alpha_max_window_vs_beta.png"):
    if not HAS_MPL:
        return
    plt.figure(figsize=(8,5))
    for i, theta in enumerate(theta_list):
        for j, gamma in enumerate(gamma_list):
            label = fr"$\theta={theta:.2f},\ \gamma={gamma:.2f}$"
            ys = [curves[(i,j)][k][1] for k in range(len(betas))]
            plt.plot(betas, ys, label=label)
    plt.axhline(ALPHA_STAR, linestyle="--", color="k", alpha=0.6, label=r"$\alpha^\* \approx 0.381966$")
    pmax,_ = argmax_delta_on_unit_half()
    plt.axvline(pmax, linestyle=":", color="k", alpha=0.6, label=fr"$p_{{\max}}\approx {pmax:.3f}$")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha_{\max}(\theta;\gamma,\beta)$")
    plt.title(r"$\alpha_{\max}$ vs. $\beta$ with floor $\gamma$ (RD-weighted equilibrium)")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.33, help="θ (weight on RD-favourable class F)")
    ap.add_argument("--gamma", type=float, default=0.16, help="γ (floor for p in F)")
    ap.add_argument("--beta",  type=float, default=0.34, help="β (ceiling for p in F)")
    ap.add_argument("--grid", action="store_true", help="Also sweep β and save CSV/plot")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory")
    args = ap.parse_args()

    # Shape of Δ and p_max
    pmax, dmax = argmax_delta_on_unit_half()
    print(f"p_max ≈ {pmax:.6f},  Delta(p_max) ≈ {dmax:.6f}")
    plot_delta_shape(pmax, dmax, path_png=f"{args.outdir}/Delta_shape.png")

    # Single evaluation
    m = m_window(args.gamma, args.beta)
    amax = alpha_max(args.theta, args.gamma, args.beta)
    print(f"[theta={args.theta:.2f}, gamma={args.gamma:.2f}, beta={args.beta:.2f}]  "
          f"m(gamma,beta)={m:.6f}   alpha_max={amax:.6f}   (alpha*={ALPHA_STAR:.6f})")

    # Optional sweep
    if args.grid:
        betas = np.linspace(0.14, 0.40, 41)  # ajusta el rango si quieres
        theta_list = [args.theta]           # puedes ampliar
        gamma_list = [args.gamma]           # puedes ampliar
        curves = {}
        for i, th in enumerate(theta_list):
            for j, gm in enumerate(gamma_list):
                rows = sweep_beta(th, gm, betas)
                curves[(i,j)] = rows
                save_csv(f"{args.outdir}/alpha_max_window_theta{th:.2f}_gamma{gm:.2f}.csv", rows)
        plot_alpha_vs_beta(curves, theta_list, gamma_list, betas,
                           path_png=f"{args.outdir}/alpha_max_window_vs_beta.png")

if __name__ == "__main__":
    main()
