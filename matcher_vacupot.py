# -*- coding: utf-8 -*-
"""
File: matcher_vacupot.py (Merged: Vacuum Alignment + Builder + Classifier Analysis)
Supports multiple molecule systems simultaneously.
Allows choosing the Global Zero reference system.
Uses WF_dpl_vcupot for vacuum calculation.
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Union
from pymatgen.io.vasp import Locpot
import matplotlib.pyplot as plt

# --- 1. Import all external tools properly ---
from builder import TrueBlochStateBuilder
from vacupot_plotter import RectAEPAWColorPlotter, PlotConfig
from classifier import StateBehaviorClassifier 

# IMPORT THE VACUUM LOGIC
from WF_dpl_vacupot import calculate_work_function

class RectangularTrueBlochMatcher:
    """
    Analyzes overlaps between pre-computed true Bloch states. 
    Performs vacuum-level alignment for physical energy consistency.
    Supports multiple molecule inputs.
    """
    def __init__(self, molecule_dirs, metal_dir, full_dir, **kwargs):
        self.molecule_dirs = list(molecule_dirs)
        self.metal_dir, self.full_dir = metal_dir, full_dir
        self.mol_labels = [os.path.basename(os.path.normpath(d)) for d in self.molecule_dirs]
        self.analysis_kwargs = kwargs
        
        # Configuration
        self.curv_tol = kwargs.get("curvature_tol", 2.5e-5)
        self.dipole_threshold = kwargs.get("dipole_threshold", 2.5)
        
        # --- Robust Band Window Handling ---
        bw_mols_input = kwargs.get("band_window_molecules", [None])
        if not isinstance(bw_mols_input, list):
            bw_mols_input = [bw_mols_input]
        if len(bw_mols_input) < len(self.molecule_dirs):
            diff = len(self.molecule_dirs) - len(bw_mols_input)
            bw_mols_input.extend([bw_mols_input[-1]] * diff)
            
        self.band_windows = {lbl: bw_mols_input[i] for i, lbl in enumerate(self.mol_labels)}
        self.band_windows.update({
            "metal": kwargs.get("band_window_metal"), 
            "full": kwargs.get("band_window_full")
        })

    @staticmethod
    def get_vacuum_potential(directory: str, curvature_tol=5e-8, dipole_threshold=0.15, min_width=10, reuse_cache=True) -> Tuple[float, bool]:
        """
        Uses WF_dpl_vcupot logic to find vacuum potential. 
        """
        cache_path = os.path.join(directory, "vacuum_potential.npz")
        
        # 1. Check Cache
        if reuse_cache and os.path.isfile(cache_path):
            try:
                data = np.load(cache_path)
                return float(data["v_vac"]), True
            except Exception:
                pass

        locpot_path = os.path.join(directory, "LOCPOT")
        if not os.path.exists(locpot_path):
            raise FileNotFoundError(f"LOCPOT missing in {directory} for vacuum alignment.")
        
        # 2. Load LOCPOT
        try:
            locpot = Locpot.from_file(locpot_path)
        except Exception as e:
             print(f"  [ERROR] Failed to load LOCPOT at {directory}: {e}")
             return 0.0, False

        # 3. Call External Logic
        # We pass ef=0.0 because we only care about the absolute vacuum potential (v_vac)
        # returned in the tuple, not the work function derived from EF.
        try:
            _, v_vac, _ = calculate_work_function(
                locpot, 
                ef=0.0, 
                curvature_tol=curvature_tol, 
                dipole_threshold=dipole_threshold, 
                min_width=min_width,
                plot=False,     # Disable plotting for batch mode
                verbose=False   # Disable verbose printing
            )
        except Exception as e:
            print(f"  [ERROR] Vacuum calculation failed for {os.path.basename(directory)}: {e}")
            # Fallback to simple max if complex logic fails
            v_vac = np.max(locpot.data['total'])

        # 4. Save and Return
        np.savez(cache_path, v_vac=v_vac)
        return float(v_vac), False

    @staticmethod
    def load_true_bloch(directory):
        path = os.path.join(directory, "true_blochstates.npz")
        if not os.path.isfile(path): return None
        return np.load(path)["psi"]

    @staticmethod
    def read_fermi_from_doscar(directory: str):
        with open(os.path.join(directory, "DOSCAR"), "r") as f:
            return float(f.readlines()[5].split()[3])

    @staticmethod
    def read_gamma_energies_from_eigenval(directory: str):
        path = os.path.join(directory, "EIGENVAL")
        with open(path, "r") as f: lines = f.readlines()
        _, nb = [int(x) for x in lines[5].split()[1:3]]
        idx = 7
        while idx < len(lines) and not lines[idx].strip(): idx += 1
        E = np.zeros((1, nb), float)
        for ib in range(nb):
            line_index = idx + 1 + ib
            if line_index < len(lines):
                E[0, ib] = float(lines[line_index].split()[1])
        return E

    def run(self, output_path: Optional[str] = None, reuse_vac_cache: bool = True, zero_reference: str = "metal"):
        """
        zero_reference: "metal", "full", or the name of a molecule (e.g. "NHC_left")
        """
        dir_map = {label: path for label, path in zip(self.mol_labels, self.molecule_dirs)}
        dir_map["metal"], dir_map["full"] = self.metal_dir, self.full_dir

        # Validate reference input
        valid_refs = ["metal", "full"] + self.mol_labels
        if zero_reference not in valid_refs:
             print(f"[ERROR] Invalid zero_reference '{zero_reference}'. Valid options: {valid_refs}. Defaulting to 'metal'.")
             zero_reference = "metal"

        # 1. Builder
        psi_arrays_full = {name: self.load_true_bloch(path) for name, path in dir_map.items()}
        if not self.analysis_kwargs.get("reuse_cached", False) or any(psi is None for psi in psi_arrays_full.values()):
            print("[MATCHER] Building missing .npz files...")
            builder = TrueBlochStateBuilder(self.molecule_dirs, self.metal_dir, self.full_dir, **self.analysis_kwargs)
            builder.build_all()
            psi_arrays_full = {name: self.load_true_bloch(path) for name, path in dir_map.items()}

        if any(psi is None for psi in psi_arrays_full.values()):
            raise FileNotFoundError("One or more required true_blochstates.npz files could not be loaded.")

        # 2. Vacuum & Fermi Detection
        print(f"\n[ALIGN] Aligning Vacuum Levels (Ref: Metal) & Setting Global Zero (Ref: {zero_reference})...")
        
        # A. Get Anchors (Metal Vacuum & Reference Fermi)
        v_vac_metal, cached_m = self.get_vacuum_potential(self.metal_dir, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        
        # Calculate the Global Shift required to zero the requested reference
        ref_path = dir_map[zero_reference]
        v_vac_ref, _ = self.get_vacuum_potential(ref_path, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        e_fermi_ref_raw = self.read_fermi_from_doscar(ref_path)
        
        # Logic for alignment
        global_shift = -(e_fermi_ref_raw + (v_vac_metal - v_vac_ref))
        print(f"  [ALIGN] Global Shift calculated: {global_shift:+.4f} eV (Ensures {zero_reference} E_f -> 0.0 eV)")

        final_fermis = {}
        homo_indices = {}

        # B. Apply to Metal
        e_f_metal_raw = self.read_fermi_from_doscar(self.metal_dir)
        e_raw_metal = self.read_gamma_energies_from_eigenval(self.metal_dir)
        
        shift_metal = global_shift 
        e_m_full = e_raw_metal + shift_metal
        final_fermis["metal"] = e_f_metal_raw + shift_metal
        
        # NEW: Proximity-based HOMO detection
        homo_indices["metal"] = np.argmin(np.abs(e_raw_metal[0] - e_f_metal_raw)) + 1
        print(f"  [METAL] V_vac={v_vac_metal:.4f} | Shift={shift_metal:+.4f} eV | Final Fermi={final_fermis['metal']:+.4f} eV")

        # C. Apply to Full System
        v_vac_full, cached_f = self.get_vacuum_potential(self.full_dir, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        e_f_full_raw = self.read_fermi_from_doscar(self.full_dir)
        
        shift_full = (v_vac_metal - v_vac_full) + global_shift
        e_f_full = self.read_gamma_energies_from_eigenval(self.full_dir) + shift_full
        final_fermis["full"] = e_f_full_raw + shift_full
        print(f"  [FULL]  V_vac={v_vac_full:.4f} | Shift={shift_full:+.4f} eV | Final Fermi={final_fermis['full']:+.4f} eV")

        # D. Apply to Molecules
        e_ms_full = []
        
        for i, md in enumerate(self.molecule_dirs):
            label = self.mol_labels[i]
            v_vac_mol, cached_mol = self.get_vacuum_potential(md, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)

            e_f_mol_raw = self.read_fermi_from_doscar(md)
            e_raw_mol = self.read_gamma_energies_from_eigenval(md)
            
            # NEW: Proximity-based HOMO detection
            homo_indices[label] = np.argmin(np.abs(e_raw_mol[0] - e_f_mol_raw)) + 1
            
            shift_mol = (v_vac_metal - v_vac_mol) + global_shift
            e_ms_full.append(e_raw_mol + shift_mol)
            
            final_fermis[label] = e_f_mol_raw + shift_mol
            print(f"  [{label}] V_vac={v_vac_mol:.4f} | Shift={shift_mol:+.4f} eV | Final Fermi={final_fermis[label]:+.4f} eV")
        
        print("[ALIGN] Alignment complete.\n")

        # 3. Slicing
        bw_f = self.band_windows.get("full")
        psi_f = psi_arrays_full["full"][bw_f, :] if bw_f else psi_arrays_full["full"]
        e_f = e_f_full[:, bw_f] if bw_f else e_f_full

        bw_m = self.band_windows.get("metal")
        psi_m = psi_arrays_full["metal"][bw_m, :] if bw_m else psi_arrays_full["metal"]
        e_m = e_m_full[:, bw_m] if bw_m else e_m_full

        psi_molecules, e_ms = [], []
        for i, lbl in enumerate(self.mol_labels):
            bw_mol = self.band_windows.get(lbl)
            psi_molecules.append(psi_arrays_full[lbl][bw_mol, :] if bw_mol else psi_arrays_full[lbl])
            e_ms.append(e_ms_full[i][:, bw_mol] if bw_mol else e_ms_full[i])

        # 4. Classifier
        S_mf = psi_m @ psi_f.conj().T
        classifier = StateBehaviorClassifier()
        E_metal_min = np.min(e_m[0])
        degenerate_indices = np.where(e_m[0] < E_metal_min + 0.05)[0]
        group_recs = []
        for metal_idx in degenerate_indices:
            for full_idx in range(e_f.shape[1]):
                dE = e_f[0, full_idx] - e_m[0, metal_idx]
                ov = np.abs(S_mf[metal_idx, full_idx])**2
                if ov > 1e-6: group_recs.append({'dE': dE, 'ov': ov})
        
        if group_recs:
            classification_info = classifier.classify_state(group_recs)
            print(f"[CLASSIFIER] Metal State Variance: {classification_info['variance']:.4f} eV^2")

        # 5. Overlaps & Write
        S_components = [psi @ psi_f.conj().T for psi in psi_molecules] + [S_mf]
        comp_labels = self.mol_labels + ["metal"]
        rows, ov_all_lines = [], []
        
        for j in range(psi_f.shape[0]):
            E_full = e_f[0, j]
            bests = {}
            for i, label in enumerate(comp_labels):
                S, energies = S_components[i], (e_ms[i] if label != "metal" else e_m)[0]
                mags = np.abs(S[:, j])**2
                i_best, ov_best, w_span = np.argmax(mags), mags.max(), mags.sum()
                E_comp = energies[i_best]
                
                bests[label] = dict(idx=i_best + 1, E=E_comp, dE=E_full-E_comp, ov_best=ov_best, w_span=w_span)
                for i_pair, z in enumerate(S[:, j]):
                    ov_all_lines.append((label, j+1, i_pair+1, energies[i_pair], E_full - energies[i_pair], np.abs(z)**2, w_span))
            rows.append({"full_idx":j+1, "E_full":E_full, "residual":max(0., 1.-sum(b['w_span'] for b in bests.values())), **bests})

        main_out = output_path or os.path.join(self.full_dir, "band_matches_rectangular.txt")
        all_out = os.path.join(self.full_dir, "band_matches_rectangular_all.txt")
        os.makedirs(os.path.dirname(main_out), exist_ok=True)
        
        with open(main_out, "w") as f:
            homo_str = "; ".join([f"{k}={v}" for k, v in homo_indices.items()])
            fermi_str = "; ".join([f"{k}={v:.5f}" for k, v in final_fermis.items()])
            
            # NEW: Calculate single aligned vacuum potential value
            v_aligned = v_vac_metal + global_shift
            vacuum_str = f"aligned={v_aligned:.5f}"
            
            f.write(f"# HOMOS: {homo_str}\n")
            f.write(f"# FERMIS: {fermi_str}\n")
            f.write(f"# VACUUMS: {vacuum_str}\n") # New header line
            f.write("# full_idx E_full " + " ".join([f"| {lbl}_idx {lbl}_E {lbl}_dE {lbl}_ov_best {lbl}_w_span" for lbl in comp_labels]) + " | residual\n")
            
            for r in sorted(rows, key=lambda x: x['full_idx']):
                f.write(f"{r['full_idx']:<8d} {r['E_full']:.4f} " + " ".join([f"| {b['idx']:<7d} {b['E']:.4f} {b['dE']:.4f} {b['ov_best']:.5f} {b['w_span']:.5f}" for b in [r[lbl] for lbl in comp_labels]]) + f" | {r['residual']:.5f}\n")
        
        with open(all_out, "w") as f:
            f.write("# component full_idx comp_idx E_comp dE_comp ov w_span_comp\n")
            for l in sorted(ov_all_lines, key=lambda x: (x[1], x[0])): f.write(f"{l[0]:<15s} {l[1]:<8d} {l[2]:<8d} {l[3]:.4f} {l[4]:.4f} {l[5]:.5f} {l[6]:.5f}\n")
        
        print(f"[MATCHER] Wrote vacuum-aligned analysis to '{main_out}'")

def run_match(molecule_dirs, metal_dir, full_dir, **kwargs) -> List[str]:
    matcher = RectangularTrueBlochMatcher(molecule_dirs, metal_dir, full_dir, **kwargs)
    reuse_vac_cache = kwargs.get("reuse_vac_cache", True)
    
    # Pass 'zero_reference' from kwargs, defaulting to 'metal'
    zero_reference = kwargs.get("zero_reference", "metal")
    
    matcher.run(kwargs.get("output_path"), reuse_vac_cache=reuse_vac_cache, zero_reference=zero_reference)
    main_file = kwargs.get("output_path") or os.path.join(full_dir, "band_matches_rectangular.txt")
    all_file = os.path.join(full_dir, "band_matches_rectangular_all.txt")
    return [main_file, all_file]

if __name__ == "__main__":
    run_kwargs = {
        "k_index": 1, 
        "tol_map": 1e-3, 
        "check_species": True,
        "band_window_molecules": [slice(0, 42)],
        "reuse_cached": True,       
        "reuse_vac_cache": True,    
        "curvature_tol": 5e-8, 
        "dipole_threshold": 0.15,
        
        # === CHOOSE YOUR ZERO HERE ===
        "zero_reference": "full" # Options: "metal", "full", "NHC_left" (or your molecule folder name)
    }
    
    match_files = run_match(
        molecule_dirs=[r'dir'],
        metal_dir=r'dir',
        full_dir=r'dir',
        **run_kwargs
    )
    main_match_file = match_files[0]
    print(f"[MAIN] Match process complete. Main output: '{main_match_file}'")
    
    print("\n--- Running Plotter ---")
    try:
        cfg = PlotConfig(
            cmap_name_simple="managua_r", 
            cmap_name_metal="vanimo_r",
            energy_range=(-9.5, 5), 
            shared_molecule_color=False,
            min_total_mol_wspan=0.025,
            
            # NEW COLORING PARAMETERS
            power_simple_neg=0.25,
            power_simple_pos=0.75,
            power_metal_neg=0.075,
            power_metal_pos=0.075,
            
            # COLORING MODE
            pick_primary=False, 
            
            # FERMI VISUALS
            show_local_fermi=True,
            
            # NEW: VACUUM VISUALS
            show_vacuum_line=True,      # Enable the black line
            vacuum_line_color="black",
            vacuum_line_width=2,      # Thick line as requested
            vacuum_line_style="-"
        )
        
        plotter = RectAEPAWColorPlotter(cfg)
        fig, axes = plotter.plot(main_match_file, bonding=True) 
        plt.show()
    except Exception as e:
        print(f"[WARN] Plotter call failed: {e}")
