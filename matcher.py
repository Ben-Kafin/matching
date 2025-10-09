# File: matcher.py
import os
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

# Import the newly separated builder class
from builder import TrueBlochStateBuilder

# Import the other tools as before
from plotter import RectAEPAWColorPlotter, PlotConfig

class RectangularTrueBlochMatcher:
    """
    Analyzes overlaps between pre-computed true Bloch states. Calls the
    TrueBlochStateBuilder class if the state .npz files do not exist.
    """
    def __init__(self, molecule_dirs, metal_dir, full_dir, **kwargs):
        self.molecule_dirs = list(molecule_dirs)
        self.metal_dir, self.full_dir = metal_dir, full_dir
        self.mol_labels = [os.path.basename(os.path.normpath(d)) for d in self.molecule_dirs]
        self.analysis_kwargs = kwargs
        self.band_windows = {lbl: kwargs.get("band_window_molecules", [None]*len(self.molecule_dirs))[i] for i, lbl in enumerate(self.mol_labels)}
        self.band_windows.update({"metal": kwargs.get("band_window_metal"), "full": kwargs.get("band_window_full")})

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
        for ib in range(nb): E[0, ib] = float(lines[idx+ib].split()[1])
        return E

    def run(self, output_path: Optional[str] = None):
        dir_map = {label: path for label, path in zip(self.mol_labels, self.molecule_dirs)}
        dir_map["metal"] = self.metal_dir
        dir_map["full"] = self.full_dir

        psi_arrays_full = {name: self.load_true_bloch(path) for name, path in dir_map.items()}
        
        if not self.analysis_kwargs.get("reuse_cached", False) or any(psi is None for psi in psi_arrays_full.values()):
            print("[MATCHER] Cached .npz files missing or reuse_cached=False. Invoking TrueBlochStateBuilder...")
            builder = TrueBlochStateBuilder(self.molecule_dirs, self.metal_dir, self.full_dir, **self.analysis_kwargs)
            builder.build_all()
            psi_arrays_full = {name: self.load_true_bloch(path) for name, path in dir_map.items()}

        if any(psi is None for psi in psi_arrays_full.values()):
            raise FileNotFoundError("One or more required true_blochstates.npz files could not be loaded after build.")

        # 1. Load ALL energies and apply initial Fermi shift
        e_f_full = self.read_gamma_energies_from_eigenval(self.full_dir) - self.read_fermi_from_doscar(self.full_dir)
        e_m_full = self.read_gamma_energies_from_eigenval(self.metal_dir) - self.read_fermi_from_doscar(self.metal_dir)
        e_ms_full = [self.read_gamma_energies_from_eigenval(md) for md in self.molecule_dirs]

        # 2. Apply band window slicing to wavefunctions AND energies
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

        # --- DEGENERATE-AWARE ALIGNMENT LOGIC ---
        S_mf = psi_m @ psi_f.conj().T
        
        # 1. Identify the group of near-degenerate lowest energy metal states
        degeneracy_threshold = 0.05 # Energy window in eV to consider states degenerate
        E_metal_min = np.min(e_m[0])
        degenerate_indices = np.where(e_m[0] < E_metal_min + degeneracy_threshold)[0]
        
        # 2. For each full state, calculate its alignment score
        # Score = (sum of overlaps with degenerate group) * (total metal w_span)
        # This favors states with strong total metal character AND strong coupling to the target group.
        group_match_strength = np.sum(np.abs(S_mf[degenerate_indices, :])**2, axis=0)
        total_metal_w_span = np.sum(np.abs(S_mf)**2, axis=0)
        alignment_scores = group_match_strength * total_metal_w_span
        
        # 3. Select the full state with the highest score for alignment
        idx_full_match = np.argmax(alignment_scores)
        E_full_match = e_f[0, idx_full_match]
        
        # 4. Align the selected full state to the lowest metal energy
        delta = E_full_match - E_metal_min
        e_f -= delta
        print(f"[ALIGN] Found {len(degenerate_indices)} degenerate metal states at ~{E_metal_min:.3f} eV.")
        print(f"[ALIGN] Shifted full system E by {-delta:+.3f} eV to align best representative state #{idx_full_match+1}.")
        
        # 5. Align molecules to the newly shifted full system
        ff_min = e_f[0].min()
        for i, e_s in enumerate(e_ms): e_ms[i] -= (e_s[0].min() - ff_min)
        # --- End of fix ---

        # Final overlap calculation and file writing
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

        main_output_file = output_path or os.path.join(self.full_dir, "band_matches_rectangular.txt")
        all_output_file = os.path.join(self.full_dir, "band_matches_rectangular_all.txt")
        os.makedirs(os.path.dirname(main_output_file), exist_ok=True)
        with open(main_output_file, "w") as f:
            f.write("# full_idx E_full " + " ".join([f"| {lbl}_idx {lbl}_E {lbl}_dE {lbl}_ov_best {lbl}_w_span" for lbl in comp_labels]) + " | residual\n")
            for r in sorted(rows, key=lambda x: x['full_idx']):
                f.write(f"{r['full_idx']:<8d} {r['E_full']:.4f} " + " ".join([f"| {b['idx']:<7d} {b['E']:.4f} {b['dE']:.4f} {b['ov_best']:.5f} {b['w_span']:.5f}" for b in [r[lbl] for lbl in comp_labels]]) + f" | {r['residual']:.5f}\n")
        with open(all_output_file, "w") as f:
            f.write("# component full_idx comp_idx E_comp dE_comp ov w_span_comp\n")
            for line in sorted(ov_all_lines, key=lambda x: (x[1], x[0])): f.write(f"{line[0]:<15s} {line[1]:<8d} {line[2]:<8d} {line[3]:.4f} {line[4]:.4f} {line[5]:.5f} {line[6]:.5f}\n")
        print(f"[MATCHER] Wrote main analysis to '{main_output_file}'\n[MATCHER] Wrote all overlaps to '{all_output_file}'")

def run_match(molecule_dirs, metal_dir, full_dir, **kwargs) -> List[str]:
    matcher = RectangularTrueBlochMatcher(molecule_dirs, metal_dir, full_dir, **kwargs)
    matcher.run(kwargs.get("output_path"))
    main_file = kwargs.get("output_path") or os.path.join(full_dir, "band_matches_rectangular.txt")
    all_file = os.path.join(full_dir, "band_matches_rectangular_all.txt")
    return [main_file, all_file]

if __name__ == "__main__":
    run_kwargs = {
        "k_index": 1, "tol_map": 1e-3, "check_species": True,
        "band_window_molecules": [slice(0, 42), slice(0, 42)],
        "reuse_cached": True
    }
    match_files = run_match(
        molecule_dirs=[r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au/NHC_left/', r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au/NHC_right/'],
        metal_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au/lone_adatom',
        full_dir=r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au/NHC2Au_complex',
        **run_kwargs
    )
    main_match_file = match_files[0]
    print(f"[MAIN] Match process complete. Main output: '{main_match_file}'")
    
    print("\n--- Running Plotter ---")
    try:
        cfg = PlotConfig(
            cmap_name_simple="managua_r", cmap_name_metal="vanimo_r",
            power_simple_neg=0.25, power_simple_pos=0.75,
            energy_range=(-25, 10), shared_molecule_color=True,
            min_total_mol_wspan=0.1
        )
        plotter = RectAEPAWColorPlotter(cfg)
        fig, axes = plotter.plot(main_match_file, bonding=True)
        plt.show()
        print(f"[MAIN] Generated and displayed color plot from '{main_match_file}'")
    except Exception as e:
        print(f"[WARN] Plotter call failed: {e}")