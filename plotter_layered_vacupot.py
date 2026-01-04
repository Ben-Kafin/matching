# File: plotter_layered.py
# -*- coding: utf-8 -*-
"""
Final, streamlined plotter with multiple coloring modes and dynamic features.
Updated with robust cursor lookups and corrected Z-ordering (Large dE on top).
"""
from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

from classifier import StateBehaviorClassifier


def _read_rect_txt_delimited(path: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"homos": {}}
    
    if not os.path.isfile(path):
        return {"rows": [], "meta": meta}
    
    with open(path, "r") as f:
        lines = f.readlines()

    header_line = ""
    for line in lines:
        s = line.strip()
        if s.startswith("# HOMOS:"):
            parts = s.replace("# HOMOS:", "").strip().split(";")
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    try:
                        meta["homos"][k.strip()] = int(v.strip())
                    except ValueError:
                        pass
        elif s.startswith("# full_idx"):
            header_line = s
            break
    
    if not header_line:
        return {"rows": [], "meta": meta}

    header_blocks = [b.strip() for b in header_line.lstrip('# ').split("|")]
    component_labels = []
    for block in header_blocks[1:-1]:
        label = block.split('_idx')[0].strip()
        component_labels.append(label)

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        
        blocks = [b.strip() for b in s.split("|")]
        if len(blocks) != len(header_blocks): continue

        try:
            full_fields = blocks[0].split()
            rec = {"full_idx": int(full_fields[0]), "E_full": float(full_fields[1])}
            
            for i, label in enumerate(component_labels):
                comp_fields = blocks[i + 1].split()
                rec[label] = {
                    "idx": int(comp_fields[0]), "E": float(comp_fields[1]),
                    "dE": float(comp_fields[2]), "ov_best": float(comp_fields[3]),
                    "w_span": float(comp_fields[4])
                }
            rec["residual"] = float(blocks[-1])
            rows.append(rec)
        except (ValueError, IndexError):
            continue
            
    return {"rows": rows, "meta": meta}


def _read_ov_all(path: str) -> Tuple[Dict[int, Dict[str, List[Dict[str, float]]]], Dict[str, List[Tuple[int, float]]]]:
    by_full = defaultdict(lambda: defaultdict(list))
    comp_idx_E = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 7: continue
            comp = parts[0]
            try:
                full_idx, comp_idx = int(parts[1]), int(parts[2])
                E_comp, dE_comp = float(parts[3]), float(parts[4])
                ov, w_span = float(parts[5]), float(parts[6])
            except Exception: continue
            rec = dict(comp_idx=comp_idx, E=E_comp, dE=dE_comp, ov=ov, w_span=w_span)
            by_full[full_idx][comp].append(rec)
            comp_idx_E[comp].setdefault(comp_idx, E_comp)
    comp_pairs = {comp: sorted(d.items(), key=lambda x: x[1]) for comp, d in comp_idx_E.items()}
    return by_full, comp_pairs


@dataclass
class PlotConfig:
    cmap_name_simple: str = "managua_r"
    cmap_name_metal: str = "vanimo_r"
    power_simple_neg: float = 0.25
    power_simple_pos: float = 0.75
    power_metal_neg: float = 0.075
    power_metal_pos: float = 0.075
    figsize: Tuple[float, float] = (8.0, 3.0)
    lw_stick: float = 2.0
    xlabel: str = "Energy (eV)"
    ylabel: str = "Normalized"
    show_fermi_line: bool = True
    fermi_line_style: str = ":"
    fermi_line_color: str = "k"
    annotate_on_hover: bool = True
    interactive: bool = True
    shared_molecule_color: bool = False
    energy_range: Optional[Tuple[float, float]] = None
    title_full: str = "Full system"
    pick_primary: Any = False 
    min_total_mol_wspan: float = 0.05


class RectAEPAWColorPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.cfg = config or PlotConfig()
        self._artists_by_comp: Dict[str, List[Any]] = {}
        # Changed: Use dictionaries for robust artist->text mapping
        self._hover_map_comp: Dict[str, Dict[Any, str]] = {} 
        self._cursor_by_comp: Dict[str, Any] = {}
        
        self._artists_f: List[Any] = []
        self._hover_map_f: Dict[Any, str] = {} # Changed: direct mapping for full plot
        self._cursor_f = None

    def _get_cmap(self, name: str):
        try: return plt.get_cmap(name)
        except Exception: return plt.get_cmap("viridis")

    def _build_colors_rank_pivot(self, pairs: List[Tuple[int, float]], cmap_name: str,
                                 center_idx: Optional[int], power_neg: float, power_pos: float) -> Dict[int, Tuple]:
        if not pairs: return {}
        ordered = sorted(pairs, key=lambda t: t[1])
        idxs = [idx for idx, _ in ordered]
        n = len(ordered)
        
        if center_idx is not None and center_idx in idxs:
            pivot = idxs.index(center_idx)
        else:
            pivot = n // 2
            
        neg_count = max(pivot, 1); pos_count = max(n - pivot - 1, 1)
        cmap = self._get_cmap(cmap_name)
        def warp(r): return abs(r) ** (power_neg if r < 0 else power_pos)
        colors = {}
        for i, (idx, _) in enumerate(ordered):
            r = (i - pivot) / neg_count if i <= pivot else (i - pivot) / pos_count
            v = 0.5 + 0.5 * np.sign(r) * warp(r)
            colors[idx] = cmap(np.clip(v, 0.0, 1.0))
        return colors

    def _mix_component_color(self, recs: List[Dict[str, float]], base_colors: Dict[int, Any],
                             default=(0.4, 0.4, 0.4, 1.0)):
        if not recs: return default
        num = np.zeros(3); denom = 0.0
        for r in recs:
            idx, w = int(r["comp_idx"]), max(float(r.get("ov", 0.0)), 0.0)
            c = base_colors.get(idx, default)
            num[:3] += w * np.array(c[:3], dtype=float)
            denom += w
        if denom <= 0: return default
        return (*(num / denom), 1.0)

    def load(self, path: str):
        return _read_rect_txt_delimited(path)

    def plot(self, path: str, ax: Optional[plt.Axes] = None, bonding: bool = False):
        data = self.load(path)
        rows = data["rows"]
        known_homos = data.get("meta", {}).get("homos", {})
        
        if not rows: print("Warning: No valid data rows parsed. Full system plot will be empty.")

        ov_all_path = os.path.join(os.path.dirname(path), "band_matches_rectangular_all.txt")
        by_full, comp_pairs = _read_ov_all(ov_all_path) if os.path.isfile(ov_all_path) else ({}, {})

        classifier = StateBehaviorClassifier()
        output_dir = os.path.dirname(path) or "."
        class_maps, class_maps_bonding = classifier.classify_and_write_summaries(by_full, output_dir)
        component_class_maps = class_maps_bonding if bonding else class_maps
        
        comp_labels_all = list(comp_pairs.keys())
        metal_present = any(lbl.lower() == "metal" for lbl in comp_labels_all)
        mol_labels = [lbl for lbl in comp_labels_all if lbl.lower() != "metal"]
        
        component_colors: Dict[str, Dict[int, Tuple]] = {}
        
        # --- COLOR MAPPING ---
        if self.cfg.shared_molecule_color and mol_labels:
            ref_label = mol_labels[0]
            center_idx = known_homos.get(ref_label)
            if center_idx is None:
                shared_map = {int(idx): float(E) for lbl in mol_labels for idx, E in comp_pairs.get(lbl, [])}
                shared_pairs = sorted(shared_map.items(), key=lambda t: t[1])
                occupied = [(idx, E) for idx, E in shared_pairs if E <= 0]
                center_idx = max(occupied, key=lambda item: item[1])[0] if occupied else None
            else:
                print(f"[PLOT] Using stored HOMO for shared molecule color: {ref_label} -> index {center_idx}")
                shared_map = {int(idx): float(E) for lbl in mol_labels for idx, E in comp_pairs.get(lbl, [])}
                shared_pairs = sorted(shared_map.items(), key=lambda t: t[1])

            shared_colors = self._build_colors_rank_pivot(
                shared_pairs, self.cfg.cmap_name_simple, center_idx, 
                self.cfg.power_simple_neg, self.cfg.power_simple_pos
            )
            for lbl in mol_labels: component_colors[lbl] = shared_colors
        else:
            for lbl in mol_labels:
                pairs = comp_pairs.get(lbl, [])
                center_idx = known_homos.get(lbl)
                if center_idx is None:
                    occupied = [(idx, E) for idx, E in pairs if E <= 0]
                    center_idx = max(occupied, key=lambda item: item[1])[0] if occupied else None
                else:
                    print(f"[PLOT] Using stored HOMO for '{lbl}': index {center_idx}")

                component_colors[lbl] = self._build_colors_rank_pivot(
                    pairs, self.cfg.cmap_name_simple, center_idx, 
                    self.cfg.power_simple_neg, self.cfg.power_simple_pos
                )

        if metal_present:
            pairs = comp_pairs.get("metal", [])
            center_idx = known_homos.get("metal")
            if center_idx is None:
                occupied = [(idx, E) for idx, E in pairs if E <= 0]
                center_idx = max(occupied, key=lambda item: item[1])[0] if occupied else None
            else:
                print(f"[PLOT] Using stored HOMO for 'metal': index {center_idx}")

            component_colors["metal"] = self._build_colors_rank_pivot(
                pairs, self.cfg.cmap_name_metal, center_idx, 
                self.cfg.power_metal_neg, self.cfg.power_metal_pos
            )

        n_comp_axes = len(mol_labels) + (1 if metal_present else 0)
        total_rows = max(1, n_comp_axes) + 1
        figsize = (self.cfg.figsize[0], max(3, 1.5 * total_rows))
        fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=figsize)
        axes = [axes] if total_rows == 1 else list(axes)
        comp_axes, ax_f = axes[:n_comp_axes], axes[-1]

        comp_iter_order = (["metal"] if metal_present else []) + mol_labels
        for i, comp_label in enumerate(comp_iter_order):
            axc = comp_axes[i]
            if self.cfg.show_fermi_line: axc.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, alpha=0.7)
            pairs, colors_map, class_map = comp_pairs.get(comp_label, []), component_colors.get(comp_label, {}), component_class_maps.get(comp_label, {})
            
            artists = self._artists_by_comp.setdefault(comp_label, [])
            hover_map = self._hover_map_comp.setdefault(comp_label, {})
            artists.clear(); hover_map.clear()
            
            for comp_idx, E in pairs:
                line = axc.vlines(E, 0, 1, color=colors_map.get(comp_idx, "black"), linewidth=self.cfg.lw_stick)
                artists.append(line)
                info = class_map.get(comp_idx, {})
                ms, var = info.get('mean_shift', 0.0), info.get('variance', 0.0)
                Ep, Ip, Ez, Iz, Em, Im = info.get("E_plus", 0.0), info.get("I_plus", 0.0), info.get("E_zero", 0.0), info.get("I_zero", 0.0), info.get("E_minus", 0.0), info.get("I_minus", 0.0)
                body = (f"up    E={Ep:+.3f}, I={Ip:.3f}\n" f"zero  E={Ez:+.3f}, I={Iz:.3f}\n" f"down  E={Em:+.3f}, I={Im:.3f}\n"
                        f"mean shift {ms:+.3f} eV\n" f"variance   {var:.3f} eV^2")
                hover_map[line] = f"{comp_label} band {comp_idx}, E {E:+.3f} eV\n{body}"
                
            axc.set_ylabel(self.cfg.ylabel); axc.set_title(comp_label)
            
            # --- Robust Component Cursor ---
            if self.cfg.annotate_on_hover and HAS_MPLCURSORS and artists:
                cur = mplcursors.cursor(artists, hover=True)
                self._cursor_by_comp[comp_label] = cur
                @cur.connect("add")
                def _on_add_comp(sel, hmap=hover_map):
                    txt = hmap.get(sel.artist)
                    if txt: sel.annotation.set_text(txt)
        
        if self.cfg.show_fermi_line: ax_f.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, alpha=0.7)
        
        metal_states_to_plot = []
        molecule_states_to_plot = []

        for rec in rows:
            E_full, full_idx = float(rec["E_full"]), int(rec["full_idx"])
            comps = by_full.get(full_idx, {})
            mode = self.cfg.pick_primary

            all_wspans = {lbl: rec.get(lbl, {}).get('w_span', 0.0) for lbl in comp_iter_order}
            total_mol_wspan = sum(w for lbl, w in all_wspans.items() if lbl != "metal")
            default_metal_color = self._mix_component_color(comps.get("metal", []), component_colors.get("metal", {}))

            comp_lines = []
            for lbl in comp_iter_order:
                top = rec.get(lbl)
                if top:
                    comp_lines.append(f"{lbl}: idx {int(top['idx'])}, E {top['E']:+.3f}, dE {top['dE']:+.3f}, ov {top['ov_best']:.4f}, w_span {top['w_span']:.4f}")
            hover_text = f"full_idx {full_idx}\nE_full {E_full:+.3f}\n" + "\n".join(comp_lines) + f"\nresidual {rec.get('residual',0.0):.5f}"

            # --- Extract plot data ---
            if mode is True:
                winner_label = max(all_wspans, key=all_wspans.get)
                dE = rec.get(winner_label, {}).get('dE', 0.0)
                color = default_metal_color
                if winner_label != "metal":
                    winner_rec = rec.get(winner_label, {})
                    if winner_idx := winner_rec.get('idx'):
                        color = component_colors.get(winner_label, {}).get(winner_idx, "black")
                plot_data = color
                plot_type = 'single'
                if winner_label == "metal":
                    metal_states_to_plot.append((abs(dE), E_full, plot_data, hover_text, plot_type))
                else:
                    molecule_states_to_plot.append((abs(dE), E_full, plot_data, hover_text, plot_type))
            
            elif mode is False:
                if total_mol_wspan >= self.cfg.min_total_mol_wspan:
                    winner_mol_label = ""
                    if mol_labels:
                         winner_mol_label = max(mol_labels, key=lambda lbl: all_wspans.get(lbl, 0.0))
                    dE = rec.get(winner_mol_label, {}).get('dE', 0.0)

                    molecule_contributions = []
                    for mol_label in mol_labels:
                        mol_rec = comps.get(mol_label)
                        if not mol_rec: continue
                        top_match = max(mol_rec, key=lambda r: r.get("ov", 0.0))
                        overlap = top_match.get("ov", 0.0)
                        color = component_colors.get(mol_label, {}).get(int(top_match["comp_idx"]))
                        if overlap > 1e-6 and color:
                            molecule_contributions.append({'overlap': overlap, 'color': color})
                    
                    molecule_contributions.sort(key=lambda x: x['overlap'], reverse=True)
                    segments = []
                    if molecule_contributions:
                        num_segments = len(molecule_contributions)
                        segment_height = 1.0 / num_segments
                        for i, contrib in enumerate(molecule_contributions):
                            y_top, y_bottom = 1.0 - (i * segment_height), 1.0 - ((i + 1) * segment_height)
                            segments.append((y_bottom, y_top, contrib['color']))
                    
                    plot_data = segments
                    plot_type = 'multi'
                    molecule_states_to_plot.append((abs(dE), E_full, plot_data, hover_text, plot_type))

                else:
                    dE = rec.get("metal", {}).get('dE', 0.0)
                    plot_data = default_metal_color
                    plot_type = 'single'
                    metal_states_to_plot.append((abs(dE), E_full, plot_data, hover_text, plot_type))

            elif mode == "blended":
                winner_label = max(all_wspans, key=all_wspans.get)
                dE = rec.get(winner_label, {}).get('dE', 0.0)
                final_rgb, total_ov_sum = np.zeros(3), 0.0
                for label in comp_iter_order:
                    recs_for_component = comps.get(label, [])
                    base_colors = component_colors.get(label, {})
                    for r in recs_for_component:
                        ov = r.get('ov', 0.0)
                        if ov > 0:
                            color = base_colors.get(int(r['comp_idx']), (0.4, 0.4, 0.4, 1.0))
                            final_rgb += np.array(color[:3]) * ov
                            total_ov_sum += ov
                blend_color = tuple(final_rgb / total_ov_sum) if total_ov_sum > 0 else (0.4, 0.4, 0.4)
                plot_data = blend_color
                plot_type = 'single'
                if winner_label == "metal":
                    metal_states_to_plot.append((abs(dE), E_full, plot_data, hover_text, plot_type))
                else:
                    molecule_states_to_plot.append((abs(dE), E_full, plot_data, hover_text, plot_type))
            
            else:
                raise ValueError("pick_primary must be True, False, or 'blended'")

        # --- SORTING LOGIC ---
        # Sort ascending (0 -> Max).
        # Small shifts (background) plotted first.
        # Large shifts (most involved) plotted last (on top).
        metal_states_to_plot.sort(key=lambda x: x[0])
        molecule_states_to_plot.sort(key=lambda x: x[0])

        self._artists_f.clear()
        self._hover_map_f.clear() # Reset full plot map
        
        all_states_to_plot = metal_states_to_plot + molecule_states_to_plot
        
        for abs_dE, E_full, plot_data, hover_text, plot_type in all_states_to_plot:
            if plot_type == 'single':
                color = plot_data
                line = ax_f.vlines(E_full, 0, 1, color=color, lw=self.cfg.lw_stick)
                self._artists_f.append(line)
                self._hover_map_f[line] = hover_text
            
            elif plot_type == 'multi':
                segments = plot_data
                if not segments:
                    line = ax_f.vlines(E_full, 0, 1, color=default_metal_color, lw=self.cfg.lw_stick)
                    self._artists_f.append(line)
                    self._hover_map_f[line] = hover_text
                    continue

                for y_bottom, y_top, color in segments:
                    line = ax_f.vlines(E_full, y_bottom, y_top, color=color, lw=self.cfg.lw_stick)
                    self._artists_f.append(line)
                    self._hover_map_f[line] = hover_text

        ax_f.set_title(self.cfg.title_full); ax_f.set_ylabel(self.cfg.ylabel); ax_f.set_xlabel(self.cfg.xlabel)
        if self.cfg.energy_range: ax_f.set_xlim(self.cfg.energy_range)
        
        # --- Robust Full Cursor ---
        if self.cfg.annotate_on_hover and HAS_MPLCURSORS and self._artists_f:
            self._cursor_f = mplcursors.cursor(self._artists_f, hover=True)
            @self._cursor_f.connect("add")
            def _on_add_f(sel, hmap=self._hover_map_f):
                txt = hmap.get(sel.artist)
                if txt: sel.annotation.set_text(txt)
        
        fig.tight_layout()
        return fig, axes


if __name__ == "__main__":
    print("\n--- Running Plotter Directly ---")
    FULL_DIR = r'C:/Users/Benjamin Kafin/Documents/VASP/lone/NHC2Au'
    main_match_file = os.path.join(FULL_DIR, "band_matches_rectangular.txt")
    
    if not os.path.isfile(main_match_file):
        print(f"[ERROR] Match file not found at: {main_match_file}")
    else:
        try:
            cfg = PlotConfig(
                cmap_name_simple="managua_r", 
                cmap_name_metal="vanimo_r",
                power_simple_neg=0.25, 
                power_simple_pos=0.75,
                energy_range=(-9.5, 3.2), 
                shared_molecule_color=True,
                min_total_mol_wspan=0.025,
                pick_primary=False 
            )
            print(f"Loading data from: {main_match_file}")
            plotter = RectAEPAWColorPlotter(cfg)
            fig, axes = plotter.plot(main_match_file, bonding=True)
            plt.show()
        except Exception as e:
            print(f"[ERROR] Plotter call failed: {e}")
            raise