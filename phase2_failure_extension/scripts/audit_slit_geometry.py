"""Read-only geometry audit and actual-frame views of the 20-degree extension.

Writes a fresh dated artifact; never changes simulation inputs or trajectories.
Displayed bonds are geometric neighbors at r < 2 A, not bond-order values.
"""
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

ROOT = Path(__file__).resolve().parents[1]
EV = 16.02176634


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'results' / ('geometry_audit_' + stamp)
    out.mkdir(exist_ok=False)
    audit = {'created_utc': stamp, 'completed_slit_trajectories': [], 'sources_sha256': {}}

    def hash_source(p):
        audit['sources_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

    for p in sorted((ROOT / 'runs').glob('ext__*slit*/trajectory.npz')):
        with np.load(p) as t:
            R, C = t['positions'], t['cells']
            row = {'case_id': p.parent.name, 'frames': len(R), 'atoms': R.shape[1],
                   'minimum_z_A': float(R[..., 2].min()), 'maximum_z_A': float(R[..., 2].max()),
                   'maximum_frame_z_span_A': float(np.ptp(R[..., 2], axis=-1).max()),
                   'cell_z_A': np.unique(C[:, 2, 2]).tolist(),
                   'maximum_cell_xy_A': float(np.abs(C[:, 0, 1]).max())}
            audit['completed_slit_trajectories'].append(row)
        hash_source(p)

    run = ROOT / 'runs/ext__S7_slit_20deg'
    record = json.loads((run / 'record.json').read_text())
    parent = ROOT / 'inputs/baseline/runs' / record['parent_run_id']
    for name in ['record.json', 'initial.extxyz', 'relaxed.extxyz', 'final.extxyz']:
        hash_source(run / name)
    original = json.loads((parent / 'record.json').read_text())
    hash_source(parent / 'record.json')
    audit['parent_boundary_conditions'] = original['boundary_conditions']
    audit['engine'] = record['engine']
    for name in ['initial.extxyz', 'final.extxyz']:
        a = read(parent / name)
        audit['parent_' + name] = {'pbc': a.pbc.tolist(), 'z_span_A': float(np.ptp(a.positions[:, 2])),
                                 'cell_A': a.cell.array.tolist()}
        hash_source(parent / name)
    audit['selected_frames'] = []
    t = np.load(run / 'trajectory.npz')
    indices = [0, 35, int(np.argmax(t['sigma_xx'])), len(t['eps_x']) - 1]
    titles = ['Relaxed reference', 'Original stopping rule', 'Later recorded peak', 'Extended endpoint']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.8))
    for ax, idx, title, letter in zip(axes.flat, indices, titles, 'abcd'):
        C = t['cells'][idx].astype(float)
        atoms = Atoms('C' * len(t['positions'][idx]), positions=t['positions'][idx], cell=C, pbc=[True, True, False])
        atoms.wrap()
        R = atoms.positions
        i, j, D = neighbor_list('ijD', atoms, 2.0)
        keep = i < j
        seg = np.stack([R[i[keep], :2], R[i[keep], :2] + D[keep, :2]], axis=1)
        # Draw translated copies so both sides of every periodic boundary bond appear.
        segs = np.concatenate([seg + [sx * C[0, 0], sy * C[1, 1]]
                               for sx in [-1, 0, 1] for sy in [-1, 0, 1]]) / 10
        rect = Rectangle((0, 0), C[0, 0] / 10, C[1, 1] / 10, facecolor='none',
                         edgecolor='#3182bd', linewidth=1.1, linestyle='--')
        ax.add_patch(rect)
        lc = LineCollection(segs, colors='#252525', linewidths=.32)
        ax.add_collection(lc)
        lc.set_clip_path(rect)
        ax.set(xlim=(-.2, 19), ylim=(-.2, 10.2), aspect='equal', xlabel='x (nm)', ylabel='y (nm)')
        stress = float(t['sigma_xx'][idx] * EV)
        ax.set_title(f'{letter}  {title}\nstrain {t["eps_x"][idx]:.3f}; stress {stress:.2f} N/m', loc='left', fontsize=10.5)
        ax.spines[['top', 'right']].set_visible(False)
        row = {'index': idx, 'strain': float(t['eps_x'][idx]), 'stress_N_m': stress,
               'shear_stress_N_m': float(t['sigma_xy'][idx] * EV),
               'driver_x_spanning': bool(t['spanning'][idx]),
               'largest_fragment_fraction': float(t['largest_fragment'][idx]),
               'n_fragments': int(t['n_fragments'][idx]),
               'max_force_eV_A': float(t['max_force_eV_A'][idx]),
               'transverse_stress_N_m': float(t['sigma_yy'][idx] * EV)}
        audit['selected_frames'].append(row)
    fig.suptitle('20° slit array: actual configurations along the planar periodic trajectory', fontsize=13, y=.96)
    fig.text(.5, .055, 'Dashed box: periodic cell in x and y; loading along x. All atoms remain at z = 10 Å.\n'
             'Same physical scale in every panel. Geometric bonds < 2 Å; residual convergence limitations apply.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.065, right=.99, bottom=.16, top=.86, hspace=.55, wspace=.15)
    for ext in ['png', 'svg']:
        fig.savefig(out / ('20deg_actual_configurations.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)
    for name in ['code/atomistics/structures/graphene.py', 'code/atomistics/structures/design_space.py',
                 'code/simulation/aqs/aqs.py', 'code/simulation/minimization/fire.py',
                 'code/potentials/rebo2scr/pytorch/torch_rebo2scr.py', 'scripts/audit_slit_geometry.py']:
        hash_source(ROOT / name)
    audit['interpretation'] = {
        'geometry': 'One atomic layer; in-plane periodic; no periodicity along z. No layer sliding.',
        'kinematics': 'Athermal strain increments in x; transverse y stress targeted to zero; no cell shear relaxation.',
        'planarity': 'The FIRE update permits all three coordinates, but exactly planar initialization and deterministic loading have retained an exactly planar trajectory. Out-of-plane stability was not tested.',
        'connectivity': 'The 20-degree recorded path retains driver x-spanning at the original stop, later peak, and extended endpoint. This is a geometric diagnostic, not proof of mechanical stability.',
        'scope': 'Neither free-edge/gripped specimens nor perturbed out-of-plane trajectories were simulated by this audit. The current extension study retains the original geometry and boundary conditions.',
        'potential': 'The configured screened REBO implementation disables the dihedral term; an out-of-plane validation should assess the force field as well as perturbations and boundary conditions.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (out / 'audit_slit_geometry.py').write_text(Path(__file__).read_text())
    print(out)


if __name__ == '__main__':
    main()
