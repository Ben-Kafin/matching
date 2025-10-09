# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:18:18 2025

@author: Benjamin Kafin
"""

# File: builder.py
import os
import numpy as np
from typing import Dict, List, Optional, Sequence
from scipy.sparse import issparse, identity, coo_matrix
from scipy.optimize import linear_sum_assignment
from ase.io import read as ase_read
from joblib import Parallel, delayed
from collections import defaultdict

from vaspwfc import vaspwfc
from aewfc import vasp_ae_wfc

class SystemData:
    """A simple data container for system properties."""
    def __init__(self, name: str, directory: str):
        self.name = name
        self.directory = directory
        self.ps: vaspwfc = None
        self.ae: vasp_ae_wfc = None
        self.atoms = None
        self.C_by_k: Dict[int, np.ndarray] = {}
        self.kpoints: List[int] = []
        self.nspins: int = 1
        self.gamma_energies: Optional[np.ndarray] = None
        self.ch_per_atom: List[int] = []
        self.nproj_total: int = 0

class TrueBlochStateBuilder:
    """
    Handles the computationally expensive task of building and saving
    true Bloch state wavefunctions (.npz files) from VASP outputs.
    """
    def __init__(self, molecule_dirs, metal_dir, full_dir, **kwargs):
        self.molecule_dirs = molecule_dirs
        self.metal_dir = metal_dir
        self.full_dir = full_dir
        self.k_index = kwargs.get("k_index", 1)
        self.tol_map = kwargs.get("tol_map", 1e-3)
        self.check_species = kwargs.get("check_species", True)

    def build_all(self):
        print("[BUILDER] Starting generation of true Bloch states...")
        
        # Load all systems
        mol_systems = [self.load_system(md, os.path.basename(os.path.normpath(md))) for md in self.molecule_dirs]
        metal_system = self.load_system(self.metal_dir, "metal")
        full_system = self.load_system(self.full_dir, "full")

        # Map atoms and build injection matrices (T)
        mol_maps = [self.map_atoms_by_coords(s.atoms, full_system.atoms) for s in mol_systems]
        metal_map = self.map_atoms_by_coords(metal_system.atoms, full_system.atoms)

        mol_Ts = [self.build_T_injection(full_system.ch_per_atom, s.ch_per_atom, m) for s, m in zip(mol_systems, mol_maps)]
        metal_T = self.build_T_injection(full_system.ch_per_atom, metal_system.ch_per_atom, metal_map)
        full_T = identity(sum(full_system.ch_per_atom))

        # Build whitener (W) from the full system
        Q = 0.5 * (full_system.ae.get_qijs() + full_system.ae.get_qijs().getH())
        W, qinfo = self.build_whitener(Q)
        print(f"[BUILDER] Whitener built: rank={qinfo['rank']}")

        # Prepare jobs for parallel processing
        jobs = [(s, s.C_by_k[self.k_index], T, d) for s, T, d in zip(mol_systems, mol_Ts, self.molecule_dirs)]
        jobs.append((metal_system, metal_system.C_by_k[self.k_index], metal_T, self.metal_dir))
        jobs.append((full_system, full_system.C_by_k[self.k_index], full_T, self.full_dir))
        
        print(f"[BUILDER] Fusing PW + AE states for {len(jobs)} systems in parallel...")
        Parallel(n_jobs=len(jobs), prefer="threads")(delayed(self._process_and_save)(s, C, T, p, W) for s, C, T, p in jobs)
        print("[BUILDER] All true Bloch state .npz files have been generated.")

    def _process_and_save(self, sys_data, Cslice, T, out_dir, W):
        B_native = self.form_B_for_slice(sys_data.ae, Cslice)
        B_lifted = self.lift_B(B_native, T)
        psi, norms = self.fuse_true_bloch_rr(Cslice, B_lifted, W)
        self.save_true_bloch(out_dir, psi, norms)

    def load_system(self, directory, name):
        sys = SystemData(name, directory)
        sys.ps = vaspwfc(os.path.join(directory, "WAVECAR")); sys.nspins = sys.ps._nspin
        rows = [np.asarray(sys.ps.readBandCoeff(ispin=1, ikpt=self.k_index, iband=i, norm=False), dtype=np.complex128) for i in range(1, sys.ps._nbands + 1)]
        sys.C_by_k[self.k_index] = np.vstack(rows)
        sys.atoms = ase_read(os.path.join(directory, "POSCAR"))
        sys.ae = vasp_ae_wfc(sys.ps, poscar=os.path.join(directory, "POSCAR"), potcar=os.path.join(directory, "POTCAR"))
        sys.ch_per_atom = [sys.ae._pawpp[it].lmmax for it in sys.ae._element_idx]
        return sys

    @staticmethod
    def map_atoms_by_coords(comp_atoms, full_atoms, tol=1e-3, check_species=True):
        comp_frac, full_frac = comp_atoms.get_scaled_positions(wrap=True), full_atoms.get_scaled_positions(wrap=True)
        cell = full_atoms.cell.array
        mapping = -np.ones(len(comp_atoms), dtype=int)
        if check_species:
            comp_by, full_by = defaultdict(list), defaultdict(list)
            for i, s in enumerate(comp_atoms.get_chemical_symbols()): comp_by[s].append(i)
            for j, s in enumerate(full_atoms.get_chemical_symbols()): full_by[s].append(j)
            for s, idx_c in comp_by.items():
                idx_f = full_by.get(s, [])
                if len(idx_f) < len(idx_c): raise ValueError(f"Full system has fewer '{s}' atoms than component.")
                D = TrueBlochStateBuilder._pairwise_min_image_dists_frac(comp_frac[idx_c], full_frac[idx_f], cell)
                rows, cols = linear_sum_assignment(np.where(D > tol, 1e6, D))
                for r, c in zip(rows, cols):
                    if D[r, c] > tol: raise ValueError(f"No match found for '{s}' atom within tolerance.")
                    mapping[idx_c[r]] = idx_f[c]
        else:
             D = TrueBlochStateBuilder._pairwise_min_image_dists_frac(comp_frac, full_frac, cell)
             rows, cols = linear_sum_assignment(np.where(D > tol, 1e6, D))
             for r, c in zip(rows, cols):
                 if D[r,c] > tol: raise ValueError("No atomic match found within tolerance.")
                 mapping[r] = c
        return mapping
    
    @staticmethod
    def _pairwise_min_image_dists_frac(comp_frac, full_frac, cell):
        df = comp_frac[:, None, :] - full_frac[None, :, :]
        df -= np.round(df) # wrap delta vectors
        dcart = np.einsum("...j,ij->...i", df, cell)
        return np.linalg.norm(dcart, axis=-1)

    @staticmethod
    def build_T_injection(full_ch_per_atom, comp_ch_per_atom, atom_map_comp_to_full):
        full_ch, comp_ch = np.asarray(full_ch_per_atom), np.asarray(comp_ch_per_atom)
        off_full = np.concatenate(([0], np.cumsum(full_ch[:-1])))
        off_comp = np.concatenate(([0], np.cumsum(comp_ch[:-1])))
        rows, cols, data = [], [], []
        for a_comp, a_full in enumerate(atom_map_comp_to_full):
            if full_ch[a_full] != comp_ch[a_comp]: raise ValueError("Projector channel mismatch.")
            rf0, cf0 = int(off_full[a_full]), int(off_comp[a_comp])
            for i in range(full_ch[a_full]):
                rows.append(rf0 + i); cols.append(cf0 + i); data.append(1.0)
        return coo_matrix((data, (rows, cols)), shape=(full_ch.sum(), comp_ch.sum())).tocsr()

    @staticmethod
    def build_whitener(Q, tol=None):
        Qm = Q.toarray() if issparse(Q) else np.asarray(Q)
        Qm = 0.5 * (Qm + Qm.conj().T)
        w, U = np.linalg.eigh(Qm)
        if tol is None: tol = max(1e-10, 1e-8 * float(w.max() if w.size else 1.0))
        keep = (w > tol)
        if not np.any(keep): raise ValueError("Q matrix is not positive definite; cannot build whitener.")
        W = U[:, keep] * (1.0 / np.sqrt(w[keep]))[None, :]
        return W, {"rank": int(keep.sum())}

    @staticmethod
    def form_B_for_slice(ae, Cslice):
        def build_vec(coeff_row): return np.asarray(ae.get_beta_njk(coeff_row), dtype=np.complex128)
        B_rows = Parallel(n_jobs=-1, prefer="threads")(delayed(build_vec)(Cslice[ib, :]) for ib in range(Cslice.shape[0]))
        return np.ascontiguousarray(B_rows)

    @staticmethod
    def lift_B(B_comp, T): return B_comp @ T.conj().T

    @staticmethod
    def fuse_true_bloch_rr(C, B, W):
        B_ortho = B @ W
        psi = np.hstack([C, B_ortho])
        norms = np.sqrt(np.einsum("ij,ij->i", psi.conj(), psi).real)
        psi /= norms[:, None]
        return psi, norms
    
    @staticmethod
    def save_true_bloch(directory, psi, norms):
        np.savez_compressed(os.path.join(directory, "true_blochstates.npz"), psi=psi, norms=norms)