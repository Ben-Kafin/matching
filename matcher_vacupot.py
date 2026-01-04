# -*- coding: utf-8 -*-
"""
File: matcher_vacupot.py (Merged: Vacuum Alignment + Builder + Classifier Analysis)
"""
import os
import numpy as np
from typing import List, Optional, Tuple
from pymatgen.io.vasp import Locpot
import matplotlib.pyplot as plt

# --- 1. Import all external tools properly ---
from builder import TrueBlochStateBuilder
from plotter_layered_vaccupot import RectAEPAWColorPlotter, PlotConfig
from classifier import StateBehaviorClassifier 

class RectangularTrueBlochMatcher:
    """
    Analyzes overlaps between pre-computed true Bloch states. 
    Performs vacuum-level alignment for physical energy consistency.
    """
    def __init__(self, molecule_dirs, metal_dir, full_dir, **kwargs):
        self.molecule_dirs = list(molecule_dirs)
        self.metal_dir, self.full_dir = metal_dir, full_dir
        self.mol_labels = [os.path.basename(os.path.normpath(d)) for d in self.molecule_dirs]
        self.analysis_kwargs = kwargs
        
        # Configuration
        self.curv_tol = kwargs.get("curvature_tol", 2.5e-5)
        self.dipole_threshold = kwargs.get("dipole_threshold", 2.5)
        
        self.band_windows = {lbl: kwargs.get("band_window_molecules", [None]*len(self.molecule_dirs))[i] 
                             for i, lbl in enumerate(self.mol_labels)}
        self.band_windows.update({"metal": kwargs.get("band_window_metal"), 
                                  "full": kwargs.get("band_window_full")})

    @staticmethod
    def get_vacuum_potential(directory: str, curvature_tol=2.5e-5, dipole_threshold=2.5, reuse_cache=True) -> Tuple[float, bool]:
        """
        Calculates the vacuum potential from LOCPOT. 
        Caches the result in 'vacuum_potential.npz' in the same directory.
        
        Returns:
            Tuple[float, bool]: (vacuum_potential, is_from_cache)
        """
        cache_path = os.path.join(directory, "vacuum_potential.npz")
        
        # --- Check Cache ---
        if reuse_cache and os.path.isfile(cache_path):
            try:
                data = np.load(cache_path)
                v_vac = float(data["v_vac"])
                # Print statements removed here; returned as status bool instead
                return v_vac, True
            except Exception:
                # Silently fail on cache load and proceed to calculation
                pass

        # --- Calculation Logic ---
        locpot_path = os.path.join(directory, "LOCPOT")
        if not os.path.exists(locpot_path):
            raise FileNotFoundError(f"LOCPOT missing in {directory} for vacuum alignment.")
        
        locpot = Locpot.from_file(locpot_path)
        potential_data = locpot.data['total']
        z_potential = np.mean(np.mean(potential_data, axis=0), axis=0)
        nz = z_potential.shape[0]
        
        z_padded = np.pad(z_potential, (2, 2), mode='wrap')
        slopes_padded = np.gradient(z_padded)
        curvatures_padded = np.abs(np.gradient(slopes_padded))
        curvatures = curvatures_padded[2:-2]
        
        stable_indices = np.where(curvatures < curvature_tol)[0]
        
        vacuum_potential = 0.0
        
        if stable_indices.size == 0:
            print(f"  [WARN] No stable vacuum found in {os.path.basename(directory)}. Defaulting to max potential.")
            vacuum_potential = np.max(z_potential)
        else:
            linear_regions = []
            if stable_indices.size > 0:
                start = stable_indices[0]
                for i in range(1, len(stable_indices)):
                    if stable_indices[i] != stable_indices[i - 1] + 1:
                        end = stable_indices[i - 1] + 1
                        linear_regions.append({
                            'start': start, 'end': end, 'width': end - start, 
                            'avg_v': np.mean(z_potential[start:end]), 'is_wrapped': False
                        })
                        start = stable_indices[i]
                end = stable_indices[-1] + 1
                linear_regions.append({
                    'start': start, 'end': end, 'width': end - start, 
                    'avg_v': np.mean(z_potential[start:end]), 'is_wrapped': False
                })

            regions = []
            if len(linear_regions) > 1:
                first_r, last_r = linear_regions[0], linear_regions[-1]
                if first_r['start'] == 0 and last_r['end'] == nz:
                    total_points = first_r['width'] + last_r['width']
                    weighted_v = (first_r['avg_v'] * first_r['width'] + last_r['avg_v'] * last_r['width']) / total_points
                    wrapped_region = {
                        'start': None, 'end': None, 'width': total_points,
                        'avg_v': weighted_v, 'is_wrapped': True, 'segments': [last_r, first_r]
                    }
                    regions = [wrapped_region] + linear_regions[1:-1]
                else:
                    regions = linear_regions
            else:
                regions = linear_regions

            potentials = [r['avg_v'] for r in regions]
            potential_spread = max(potentials) - min(potentials)
            dipole_detected = potential_spread > dipole_threshold
            
            if dipole_detected:
                best_region = max(regions, key=lambda x: x['avg_v'])
                mode_str = "Dipole (Highest V)"
            else:
                best_region = max(regions, key=lambda x: x['width'])
                mode_str = "Standard (Widest)"

            if best_region['is_wrapped']:
                seg1, seg2 = best_region['segments']
                chunk1 = potential_data[:, :, seg1['start']:seg1['end']]
                chunk2 = potential_data[:, :, seg2['start']:seg2['end']]
                vacuum_potential = (np.sum(chunk1) + np.sum(chunk2)) / (chunk1.size + chunk2.size)
            else:
                v_start, v_end = best_region['start'], best_region['end']
                vacuum_potential = np.mean(potential_data[:, :, v_start:v_end])

            print(f"    -> Vac Mode: {mode_str:<18} | Spread: {potential_spread:.3f}eV | Level: {vacuum_potential:.4f}eV")

        # --- Save Cache ---
        np.savez(cache_path, v_vac=vacuum_potential)
        return vacuum_potential, False

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

    def run(self, output_path: Optional[str] = None, reuse_vac_cache: bool = True):
        dir_map = {label: path for label, path in zip(self.mol_labels, self.molecule_dirs)}
        dir_map["metal"], dir_map["full"] = self.metal_dir, self.full_dir

        # --- 1. Builder Integration ---
        psi_arrays_full = {name: self.load_true_bloch(path) for name, path in dir_map.items()}
        if not self.analysis_kwargs.get("reuse_cached", False) or any(psi is None for psi in psi_arrays_full.values()):
            print("[MATCHER] Cached .npz files missing or reuse_cached=False. Invoking TrueBlochStateBuilder...")
            builder = TrueBlochStateBuilder(self.molecule_dirs, self.metal_dir, self.full_dir, **self.analysis_kwargs)
            builder.build_all()
            psi_arrays_full = {name: self.load_true_bloch(path) for name, path in dir_map.items()}

        if any(psi is None for psi in psi_arrays_full.values()):
            raise FileNotFoundError("One or more required true_blochstates.npz files could not be loaded.")

        # --- 2. Vacuum & Fermi Detection (ANCHOR: METAL SYSTEM) ---
        print("\n[ALIGN] Detecting vacuum and Fermi levels (Reference: Metal System)...")
        
        # A. Analyze Metal (The Anchor)
        print("  [METAL] Analyzing vacuum (Anchor)...")
        v_vac_metal, cached_m = self.get_vacuum_potential(self.metal_dir, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        
        if cached_m:
            print(f"    [CACHE] Loaded vacuum potential from {os.path.basename(self.metal_dir)}/vacuum_potential.npz: {v_vac_metal:.4f} eV")
        else:
            print("    [CALC] Calculating vacuum potential from LOCPOT...")
            
        e_f_metal_raw = self.read_fermi_from_doscar(self.metal_dir)
        e_raw_metal = self.read_gamma_energies_from_eigenval(self.metal_dir)
        
        global_shift = -e_f_metal_raw
        shift_metal = global_shift 
        e_m_full = e_raw_metal + shift_metal
        
        occ_m = np.where(e_raw_metal[0] < e_f_metal_raw)[0]
        homo_idx_m = occ_m[-1] + 1 if occ_m.size > 0 else 0
        print(f"  [METAL]  Total Shift: {shift_metal:+.4f} eV | HOMO: Band {homo_idx_m} (Set to 0.0 eV)")

        # B. Analyze Full System
        print("  [FULL] Analyzing vacuum...")
        v_vac_full, cached_f = self.get_vacuum_potential(self.full_dir, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
        
        if cached_f:
             print(f"    [CACHE] Loaded vacuum potential from {os.path.basename(self.full_dir)}/vacuum_potential.npz: {v_vac_full:.4f} eV")
        else:
             print("    [CALC] Calculating vacuum potential from LOCPOT...")

        e_f_full_raw = self.read_fermi_from_doscar(self.full_dir)
        
        shift_full = (v_vac_metal - v_vac_full) + global_shift
        e_f_full = self.read_gamma_energies_from_eigenval(self.full_dir) + shift_full
        
        e_f_full_final = e_f_full_raw + shift_full
        print(f"  [FULL]   Total Shift: {shift_full:+.4f} eV | Final Fermi: {e_f_full_final:+.4f} eV (Delta Phi)")

        # C. Analyze Molecule Systems
        e_ms_full = []
        homo_indices = {"metal": homo_idx_m} # Store calculated HOMOs here
        
        for i, md in enumerate(self.molecule_dirs):
            label = self.mol_labels[i]
            print(f"  [{label: <7}] Analyzing vacuum...")
            v_vac_mol, cached_mol = self.get_vacuum_potential(md, self.curv_tol, self.dipole_threshold, reuse_cache=reuse_vac_cache)
            
            if cached_mol:
                print(f"    [CACHE] Loaded vacuum potential from {os.path.basename(md)}/vacuum_potential.npz: {v_vac_mol:.4f} eV")
            else:
                print("    [CALC] Calculating vacuum potential from LOCPOT...")

            e_f_mol_raw = self.read_fermi_from_doscar(md)
            e_raw_mol = self.read_gamma_energies_from_eigenval(md)
            
            occ_mol = np.where(e_raw_mol[0] < e_f_mol_raw)[0]
            homo_idx_mol = occ_mol[-1] + 1 if occ_mol.size > 0 else 0
            homo_indices[label] = homo_idx_mol # Store it
            
            shift_mol = (v_vac_metal - v_vac_mol) + global_shift
            e_ms_full.append(e_raw_mol + shift_mol)
            print(f"  [{label: <7}] Total Shift: {shift_mol:+.4f} eV | HOMO: Band {homo_idx_mol}")
        
        print("[ALIGN] Alignment process complete.\n")

        # --- 3. Band Window Slicing ---
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

        # --- 4. CLASSIFIER ANALYSIS ---
        S_mf = psi_m @ psi_f.conj().T
        print("[CLASSIFIER] Running post-alignment interface analysis...")
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
            print(f"  -> Metal State Variance: {classification_info['variance']:.4f} eV^2")
        else:
             print("  -> No significant overlaps found for low-energy metal states.")
        print("")

        # --- 5. Final Overlap Calculation & Write ---
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
                # Adjust index output to be 1-based absolute index, handling windows if necessary
                # Note: This logic assumes i_best is relative to the window. 
                # Ideally, we'd map back to absolute, but for now we keep 1-based relative.
                bests[label] = dict(idx=i_best + 1, E=E_comp, dE=E_full-E_comp, ov_best=ov_best, w_span=w_span)
                for i_pair, z in enumerate(S[:, j]):
                    ov_all_lines.append((label, j+1, i_pair+1, energies[i_pair], E_full - energies[i_pair], np.abs(z)**2, w_span))
            rows.append({"full_idx":j+1, "E_full":E_full, "residual":max(0., 1.-sum(b['w_span'] for b in bests.values())), **bests})

        main_out = output_path or os.path.join(self.full_dir, "band_matches_rectangular.txt")
        all_out = os.path.join(self.full_dir, "band_matches_rectangular_all.txt")
        os.makedirs(os.path.dirname(main_out), exist_ok=True)
        
        with open(main_out, "w") as f:
            # === UPDATED: Write HOMO info to header ===
            homo_str = "; ".join([f"{k}={v}" for k, v in homo_indices.items()])
            f.write(f"# HOMOS: {homo_str}\n")
            f.write("# full_idx E_full " + " ".join([f"| {lbl}_idx {lbl}_E {lbl}_dE {lbl}_ov_best {lbl}_w_span" for lbl in comp_labels]) + " | residual\n")
            for r in sorted(rows, key=lambda x: x['full_idx']):
                f.write(f"{r['full_idx']:<8d} {r['E_full']:.4f} " + " ".join([f"| {b['idx']:<7d} {b['E']:.4f} {b['dE']:.4f} {b['ov_best']:.5f} {b['w_span']:.5f}" for b in [r[lbl] for lbl in comp_labels]]) + f" | {r['residual']:.5f}\n")
        
        with open(all_out, "w") as f:
            f.write("# component full_idx comp_idx E_comp dE_comp ov w_span_comp\n")
            for l in sorted(ov_all_lines, key=lambda x: (x[1], x[0])): f.write(f"{l[0]:<15s} {l[1]:<8d} {l[2]:<8d} {l[3]:.4f} {l[4]:.4f} {l[5]:.5f} {l[6]:.5f}\n")
        
        print(f"[MATCHER] Wrote vacuum-aligned analysis to '{main_out}'")

def run_match(molecule_dirs, metal_dir, full_dir, **kwargs) -> List[str]:
    matcher = RectangularTrueBlochMatcher(molecule_dirs, metal_dir, full_dir, **kwargs)
    
    # Extract the reuse_vac_cache arg from kwargs (defaulting to True if not present)
    reuse_vac_cache = kwargs.get("reuse_vac_cache", True)
    
    matcher.run(kwargs.get("output_path"), reuse_vac_cache=reuse_vac_cache)
    main_file = kwargs.get("output_path") or os.path.join(full_dir, "band_matches_rectangular.txt")
    all_file = os.path.join(full_dir, "band_matches_rectangular_all.txt")
    return [main_file, all_file]

if __name__ == "__main__":
    run_kwargs = {
        "k_index": 1, 
        "tol_map": 1e-3, 
        "check_species": True,
        "band_window_molecules": [slice(0, 42)],
        "reuse_cached": True,       # Controls wavefunction caching (builder.py)
        "reuse_vac_cache": True,    # Controls vacuum potential caching (matcher.py)
        "curvature_tol": 2.5e-5, 
        "dipole_threshold": 2.5 
    }
    match_files = run_match(
        molecule_dirs=[r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/NHC/kp552'],
        metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/adatom_surface/kpoints552',
        full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/fcc/kp552',
        **run_kwargs
    )
    main_match_file = match_files[0]
    print(f"[MAIN] Match process complete. Main output: '{main_match_file}'")
    
    print("\n--- Running Plotter ---")
    try:
        cfg = PlotConfig(
            cmap_name_simple="managua_r", 
            cmap_name_metal="vanimo_r",
            power_simple_neg=0.25, 
            power_simple_pos=0.75,
            energy_range=(-20.0, 7.25), 
            shared_molecule_color=True,
            min_total_mol_wspan=0.02,
            pick_primary=False,
        )
        plotter = RectAEPAWColorPlotter(cfg)
        fig, axes = plotter.plot(main_match_file, bonding=True)
        plt.show()
        print(f"[MAIN] Generated and displayed color plot from '{main_match_file}'")
    except Exception as e:
        print(f"[WARN] Plotter call failed: {e}")