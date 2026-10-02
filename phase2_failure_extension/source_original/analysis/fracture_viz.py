"""Fracture visualisation: event-selected snapshot panels and MP4 movies from stored AQS trajectories.

Frames are selected by physical events (not evenly spaced):
  0 relaxed (zero strain)      1 pre-damage (last frame without broken bonds)
  2 first irreversible change  3 peak load
  4 onset of major propagation (largest single-step bond loss after the peak)
  5 representative post-peak   6 final failed configuration
plus additional frames for every discrete damage event in progressive fracture (up to a cap)."""
from __future__ import annotations
import os, sys, json, subprocess, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
EV = 16.0217663


def load_traj(npz_path):
    d = np.load(npz_path)
    T = {k: d[k] for k in d.files if not k.startswith("bonds_")}
    T["bonds"] = [d[f"bonds_{k}"] for k in range(len(T["positions"]))]
    return T


def select_event_frames(T, max_extra=4):
    sxx = T["sigma_xx"]; nb = T["n_broken_cum"]; n = len(sxx)
    ip = int(np.argmax(sxx))
    dmg = np.where(np.diff(nb, prepend=nb[0]) > 0)[0]
    first = int(dmg[0]) if len(dmg) else None
    pre = max(first - 1, 0) if first is not None else ip
    # onset of major propagation: largest bond loss per step at/after the first damage
    inc = np.diff(nb, prepend=nb[0])
    onset = int(np.argmax(inc)) if inc.max() > 0 else ip
    post = min(max(ip, onset) + 1, n - 1)
    frames = {"relaxed": 0, "pre_damage": pre, "first_damage": first if first is not None else ip, "peak_load": ip,
              "major_propagation": onset, "post_peak": post, "final": n - 1}
    # extra discrete damage events (progressive fracture)
    extra = [int(k) for k in dmg if k not in frames.values()]
    if len(extra) > max_extra:
        idx = np.linspace(0, len(extra) - 1, max_extra).round().astype(int)
        extra = [extra[i] for i in idx]
    order = ["relaxed", "pre_damage", "first_damage", "peak_load", "major_propagation", "post_peak", "final"]
    labels = {}
    for k in order:
        f = frames[k]
        labels[f] = (labels[f] + " = " + k) if f in labels else k
    for f in extra:
        if f not in labels:
            labels[f] = "damage_event"
    sel = sorted(labels.items(), key=lambda x: x[0])
    return [(lab, f) for f, lab in sel]


def atom_colors(T, k, mode, ref=None):
    if mode == "energy":
        v = T["peratom_energy"][k].astype(float); return v, "energy per atom (eV)", "viridis"
    if mode == "energy_rel":
        v = T["peratom_energy"][k].astype(float) - T["peratom_energy"][0].astype(float); return v, "energy change per atom (eV)", "magma"
    if mode == "stress":
        v = T["peratom_virial"][k][:, 0].astype(float); return v, "per-atom virial xx (eV)", "coolwarm"
    if mode == "coordination":
        v = T["coordination"][k].astype(float); return v, "coordination", "viridis"
    if mode == "displacement":
        P0, P = T["positions"][0].astype(float), T["positions"][k].astype(float); c0, c = T["cells"][0], T["cells"][k]
        aff = P0 * np.array([c[0, 0] / c0[0, 0], c[1, 1] / c0[1, 1], 1.0]); dd = P - aff
        for ax in range(2):
            dd[:, ax] -= c[ax, ax] * np.round(dd[:, ax] / c[ax, ax])
        return np.linalg.norm(dd, axis=1), "non-affine displacement (A)", "plasma"
    raise ValueError(mode)


def draw_frame(ax, T, k, mode="energy_rel", vmin=None, vmax=None, damage_atoms=None, s=6, show_bonds=True, lw=0.6):
    P = T["positions"][k].astype(float); c = T["cells"][k]
    v, label, cmap = atom_colors(T, k, mode)
    if vmin is None:
        vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-6
    if show_bonds:
        b = T["bonds"][k]
        if len(b):
            p0, p1 = P[b[:, 0], :2], P[b[:, 1], :2]
            d = p1 - p0
            for ax_ in range(2):
                d[:, ax_] -= c[ax_, ax_] * np.round(d[:, ax_] / c[ax_, ax_])
            segs = np.stack([p0, p0 + d], 1)
            ax.add_collection(LineCollection(segs, colors="0.55", linewidths=lw, zorder=1))
    sc = ax.scatter(P[:, 0], P[:, 1], c=v, s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, zorder=2)
    if damage_atoms is not None and len(damage_atoms):
        ax.scatter(P[damage_atoms, 0], P[damage_atoms, 1], s=s * 5, facecolors="none", edgecolors="red", linewidths=0.8, zorder=3)
    ax.set_xlim(-2, c[0, 0] + 2); ax.set_ylim(-2, c[1, 1] + 2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return sc, label


def damaged_atoms_until(record, eps):
    ev = record.get("broken_bond_events", [])
    at = set()
    for e in ev:
        if e[0] <= eps + 1e-9:
            at.add(int(e[1])); at.add(int(e[2]))
    return np.array(sorted(at), int)


def progression_panel(run_id, out_dir, mode="energy_rel", max_frames=9):
    from experiments import db
    rec = db.load_record(run_id)
    T = load_traj(os.path.join(ROOT, rec["trajectory"]))
    sel = select_event_frames(T)[:max_frames]
    n = len(sel)
    ncol = min(n, 5); nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.4 * ncol, 3.6 * nrow + 3.2))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.85])
    eps = T["eps_x"]; sxx = T["sigma_xx"] * EV
    vals = [atom_colors(T, f, mode)[0] for _, f in sel]
    vmin = min(np.percentile(v, 1) for v in vals); vmax = max(np.percentile(v, 99) for v in vals)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    for i, (name, f) in enumerate(sel):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        dmg = damaged_atoms_until(rec, eps[f])
        sc, label = draw_frame(ax, T, f, mode, vmin, vmax, damage_atoms=dmg)
        ax.set_title(f"{i+1}. {name.replace('_', ' ')}\n" + r"$\epsilon$" + f"={eps[f]:.4f}, " + r"$\sigma$" + f"={sxx[f]:.1f} N/m, broken={int(T['n_broken_cum'][f])}", fontsize=8)
    axs = fig.add_subplot(gs[nrow, :])
    axs.plot(eps, sxx, "k-", lw=1.5)
    for i, (name, f) in enumerate(sel):
        axs.plot(eps[f], sxx[f], "o", ms=7, color=plt.cm.tab10(i % 10)); axs.annotate(str(i + 1), (eps[f], sxx[f]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axs.set_xlabel("engineering strain (-)"); axs.set_ylabel("2D stress (N/m)"); axs.grid(alpha=0.3)
    axs.set_title(f"{rec['name']} ({rec['family']}, N={rec['n_atoms']}, phi={rec.get('porosity', 0):.2f}); frames selected by events", fontsize=9)
    cb = fig.colorbar(sc, ax=axs, orientation="vertical", fraction=0.03, pad=0.01); cb.set_label(label, fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"progression_{rec['name']}_{mode}")
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(base + "." + ext, dpi=180 if ext == "png" else None)
    plt.close(fig)
    meta = {"run_id": run_id, "name": rec["name"], "frames": [{"label": nm, "frame": int(f), "eps": float(eps[f]), "sigma_Nm": float(sxx[f]), "n_broken": int(T["n_broken_cum"][f])} for nm, f in sel], "mode": mode, "file": base + ".png"}
    json.dump(meta, open(base + ".json", "w"), indent=1)
    return meta


def before_after_panel(run_id, out_dir, mode="energy_rel"):
    from experiments import db
    rec = db.load_record(run_id)
    T = load_traj(os.path.join(ROOT, rec["trajectory"]))
    sel = dict(select_event_frames(T))
    frames = [("relaxed", 0), ("peak load", sel.get("peak_load", 0)), ("final", len(T["eps_x"]) - 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (nm, f) in zip(axes, frames):
        dmg = damaged_atoms_until(rec, T["eps_x"][f])
        sc, label = draw_frame(ax, T, f, mode, damage_atoms=dmg)
        ax.set_title(f"{nm}: eps={T['eps_x'][f]:.3f}, sigma={T['sigma_xx'][f]*EV:.1f} N/m", fontsize=9)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01, label=label)
    base = os.path.join(out_dir, f"before_after_{rec['name']}")
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(base + "." + ext, dpi=180 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return base + ".png"


def make_movie(run_id, out_path, mode="energy_rel", fps=8, max_frames=160, dpi=110):
    """Render every stored frame (subsampled to max_frames) with a synchronised stress-strain inset and encode with ffmpeg."""
    from experiments import db
    rec = db.load_record(run_id)
    T = load_traj(os.path.join(ROOT, rec["trajectory"]))
    n = len(T["eps_x"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
    # always include event frames
    for _, f in select_event_frames(T):
        idx = np.union1d(idx, [f])
    tmp = out_path + "_frames"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    eps = T["eps_x"]; sxx = T["sigma_xx"] * EV
    vals = atom_colors(T, idx[-1], mode)[0]; v0 = atom_colors(T, 0, mode)[0]
    vmin = min(np.percentile(v0, 1), np.percentile(vals, 1)); vmax = max(np.percentile(v0, 99), np.percentile(vals, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    c = T["cells"][0]; aspect = c[1, 1] / c[0, 0]
    events = {f: nm for nm, f in select_event_frames(T)}
    for m, f in enumerate(idx):
        fig = plt.figure(figsize=(11, 5.2))
        ax = fig.add_axes([0.02, 0.05, 0.5, 0.88]); axs = fig.add_axes([0.6, 0.15, 0.37, 0.75])
        dmg = damaged_atoms_until(rec, eps[f])
        sc, label = draw_frame(ax, T, f, mode, vmin, vmax, damage_atoms=dmg, s=8, lw=0.7)
        ax.set_title(f"{rec['name']}   eps = {eps[f]:.4f}   sigma = {sxx[f]:.1f} N/m   broken bonds: {int(T['n_broken_cum'][f])}" + (f"   [{events[f].replace('_',' ')}]" if f in events else ""), fontsize=9)
        axs.plot(eps, sxx, "k-", lw=1.2); axs.plot(eps[f], sxx[f], "ro", ms=8)
        for ff, nm in events.items():
            axs.axvline(eps[ff], color="0.7", lw=0.6, ls=":")
        axs.set_xlabel("strain (-)"); axs.set_ylabel("2D stress (N/m)"); axs.grid(alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01); cb.set_label(label, fontsize=8)
        fig.savefig(os.path.join(tmp, f"f{m:04d}.png"), dpi=dpi)
        plt.close(fig)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path]
    subprocess.run(cmd, check=True)
    meta = {"run_id": run_id, "frames_rendered": [int(x) for x in idx], "event_frames": {nm: int(f) for f, nm in events.items()}, "fps": fps, "mode": mode,
            "trajectory": rec["trajectory"], "movie": os.path.relpath(out_path, ROOT)}
    json.dump(meta, open(out_path.replace(".mp4", ".json"), "w"), indent=1)
    shutil.rmtree(tmp)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("run_id"); ap.add_argument("--movie", action="store_true"); ap.add_argument("--mode", default="energy_rel")
    a = ap.parse_args()
    print(progression_panel(a.run_id, os.path.join(ROOT, "figures", "progression"), a.mode))
    if a.movie:
        print(make_movie(a.run_id, os.path.join(ROOT, "movies", f"{a.run_id}.mp4"), a.mode))
