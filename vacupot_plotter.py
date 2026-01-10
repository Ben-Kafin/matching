# File: plotter_layered_vacupot.py
# -*- coding: utf-8 -*-
"""
Final, patched plotter. 
Integrated advanced rank-pivot coloring and three-way pick_primary modes from plotter.py.
Retained local Fermi plotting, metadata parsing, and layered sorting from vacupot.
"""
from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

from classifier import StateBehaviorClassifier


def _read_rect_txt_delimited(path: str) -> Dict[str, Any]:
    """
    Parses matcher output. Retains metadata support for HOMOS and FERMIS.
    """
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"homos": {}, "fermis": {}}
    
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
                    try: meta["homos"][k.strip()] = int(v.strip())
                    except ValueError: pass
        elif s.startswith("# FERMIS:"):
            parts = s.replace("# FERMIS:", "").strip().split(";")
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    try: meta["fermis"][k.strip()] = float(v.strip())
                    except ValueError: pass
        elif s.startswith("# full_idx") or s.startswith("#full_idx"):
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
        if not s or s.startswith("#"): continue
        
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
        except (ValueError, IndexError): continue
            
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
    # Adopting rank-pivot power warping from plotter.py
    power_simple_neg: float = 0.25
    power_simple_pos: float = 0.75
    power_metal_neg: float = 0.075
    power_metal_pos: float = 0.075
    figsize: Tuple[float, float] = (8.0, 3.0)
    lw_stick: float = 2.0
    xlabel: str = "Energy (eV)"
    ylabel: str = "Normalized"
    
    # Global/Local Fermi Visuals
    show_fermi_line: bool = True
    fermi_line_style: str = ":"
    fermi_line_color: str = "k"
    show_local_fermi: bool = True
    local_fermi_style: str = "--"
    local_fermi_color: str = "red"
    
    annotate_on_hover: bool = True
    interactive: bool = True
    shared_molecule_color: bool = False
    energy_range: Optional[Tuple[float, float]] = None
    title_full: str = "Full system"
    # Support for True, False, and "blended"
    pick_primary: Any = "blended" 
    min_total_mol_wspan: float = 0.025


class RectAEPAWColorPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.cfg = config or PlotConfig()
        self._artists_by_comp: Dict[str, List[Any]] = {}
        self._hover_map_comp: Dict[str, Dict[Any, str]] = {} 
        self._cursor_by_comp: Dict[str, Any] = {}
        self._artists_f: List[Any] = []
        self._hover_map_f: Dict[Any, str] = {}
        self._cursor_f = None

    def _get_cmap(self, name: str):
        try: return plt.get_cmap(name)
        except Exception: return plt.get_cmap("viridis")

    def _build_colors_rank_pivot(self, pairs: List[Tuple[int, float]], cmap_name: str,
                                 center_idx: Optional[int], power_neg: float, power_pos: float) -> Dict[int, Tuple]:
        """
        Rank-based colormap assignment with power-law warping.
        """
        if not pairs: return {}
        ordered = sorted(pairs, key=lambda t: t[1])
        idxs = [idx for idx, _ in ordered]
        n = len(ordered)
        pivot = idxs.index(center_idx) if (center_idx is not None and center_idx in idxs) else n // 2
        neg_count = max(pivot, 1); pos_count = max(n - pivot - 1, 1)
        cmap = self._get_cmap(cmap_name)
        def warp(r): return abs(r) ** (power_neg if r < 0 else power_pos)
        colors = {}
        for i, (idx, _) in enumerate(ordered):
            r = (i - pivot) / neg_count if i <= pivot else (i - pivot) / pos_count
            v = 0.5 + 0.5 * np.sign(r) * warp(r)
            colors[idx] = cmap(np.clip(v, 0.0, 1.0))
        return colors

    def _mix_component_color(self, recs: List[Dict[str, float]], base_colors: Dict[int, Any], default=(0.4, 0.4, 0.4, 1.0)):
        """
        Weighted RGB average for metal background.
        """
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
        rows, meta = data["rows"], data["meta"]
        known_homos, known_fermis = meta.get("homos", {}), meta.get("fermis", {})
        
        ov_all_path = os.path.join(os.path.dirname(path), "band_matches_rectangular_all.txt")
        by_full, comp_pairs = _read_ov_all(ov_all_path) if os.path.isfile(ov_all_path) else ({}, {})

        classifier = StateBehaviorClassifier()
        output_dir = os.path.dirname(path) or "."
        class_maps, class_maps_bonding = classifier.classify_and_write_summaries(by_full, output_dir)
        component_class_maps = class_maps_bonding if bonding else class_maps
        
        comp_labels_all = list(comp_pairs.keys())
        metal_present = "metal" in comp_labels_all
        mol_labels = [lbl for lbl in comp_labels_all if lbl.lower() != "metal"]
        
        component_colors: Dict[str, Dict[int, Tuple]] = {}
        
        # Color Centering: Metadata priority with energetic fallback
        def get_center_idx(lbl, pairs):
            if lbl in known_homos: return int(known_homos[lbl])
            occupied = [(idx, E) for idx, E in pairs if E <= 0]
            return max(occupied, key=lambda x: x[1])[0] if occupied else None

        if self.cfg.shared_molecule_color and mol_labels:
            ref_label = mol_labels[0]
            shared_map = {int(idx): float(E) for lbl in mol_labels for idx, E in comp_pairs.get(lbl, [])}
            shared_pairs = sorted(shared_map.items(), key=lambda t: t[1])
            center = get_center_idx(ref_label, shared_pairs)
            shared_colors = self._build_colors_rank_pivot(shared_pairs, self.cfg.cmap_name_simple, center, self.cfg.power_simple_neg, self.cfg.power_simple_pos)
            for lbl in mol_labels: component_colors[lbl] = shared_colors
        else:
            for lbl in mol_labels:
                pairs = comp_pairs.get(lbl, [])
                center = get_center_idx(lbl, pairs)
                component_colors[lbl] = self._build_colors_rank_pivot(pairs, self.cfg.cmap_name_simple, center, self.cfg.power_simple_neg, self.cfg.power_simple_pos)

        if metal_present:
            pairs = comp_pairs.get("metal", [])
            center = get_center_idx("metal", pairs)
            component_colors["metal"] = self._build_colors_rank_pivot(pairs, self.cfg.cmap_name_metal, center, self.cfg.power_metal_neg, self.cfg.power_metal_pos)

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
            local_E_f = known_fermis.get(comp_label)
            if self.cfg.show_local_fermi and local_E_f is not None:
                axc.axvline(local_E_f, color=self.cfg.local_fermi_color, linestyle=self.cfg.local_fermi_style, alpha=0.9)
            
            pairs, colors_map, class_map = comp_pairs.get(comp_label, []), component_colors.get(comp_label, {}), component_class_maps.get(comp_label, {})
            artists, hover_map = self._artists_by_comp.setdefault(comp_label, []), self._hover_map_comp.setdefault(comp_label, {})
            artists.clear(); hover_map.clear()
            
            for comp_idx, E in pairs:
                line = axc.vlines(E, 0, 1, color=colors_map.get(comp_idx, "black"), linewidth=self.cfg.lw_stick)
                artists.append(line)
                info = class_map.get(comp_idx, {})
                ms, var = info.get('mean_shift', 0.0), info.get('variance', 0.0)
                body = (f"up    E={info.get('E_plus',0):+.3f}, I={info.get('I_plus',0):.3f}\n"
                        f"zero  E={info.get('E_zero',0):+.3f}, I={info.get('I_zero',0):.3f}\n"
                        f"down  E={info.get('E_minus',0):+.3f}, I={info.get('I_minus',0):.3f}\n"
                        f"mean shift {ms:+.3f} eV\n" f"variance   {var:.3f} eV^2")
                hover_map[line] = f"{comp_label} band {comp_idx}, E {E:+.3f} eV\n{body}"
            axc.set_ylabel(self.cfg.ylabel); axc.set_title(comp_label)
            if self.cfg.annotate_on_hover and HAS_MPLCURSORS and artists:
                cur = mplcursors.cursor(artists, hover=True); self._cursor_by_comp[comp_label] = cur
                @cur.connect("add")
                def _on_add_comp(sel, hmap=hover_map):
                    if (txt := hmap.get(sel.artist)): sel.annotation.set_text(txt)
        
        # --- Full System Plot with Selection Modes ---
        if self.cfg.show_fermi_line: ax_f.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, alpha=0.7)
        if self.cfg.show_local_fermi and (lf_full := known_fermis.get("full")) is not None:
             ax_f.axvline(lf_full, color=self.cfg.local_fermi_color, linestyle=self.cfg.local_fermi_style, alpha=0.9)

        metal_states_to_plot, molecule_states_to_plot = [], []

        for rec in rows:
            E_full, full_idx = float(rec["E_full"]), int(rec["full_idx"])
            comps = by_full.get(full_idx, {})
            all_wspans = {lbl: rec.get(lbl, {}).get('w_span', 0.0) for lbl in comp_iter_order}
            total_mol_wspan = sum(w for lbl, w in all_wspans.items() if lbl != "metal")
            def_metal_col = self._mix_component_color(comps.get("metal", []), component_colors.get("metal", {}))

            comp_lines = [f"{lbl}: idx {int(top['idx'])}, E {top['E']:+.3f}, dE {top['dE']:+.3f}, ov {top['ov_best']:.4f}, w_span {top['w_span']:.4f}" 
                          for lbl in comp_iter_order if (top := rec.get(lbl))]
            hover_text = f"full_idx {full_idx}\nE_full {E_full:+.3f}\n" + "\n".join(comp_lines) + f"\nresidual {rec.get('residual',0.0):.5f}"

            # 1. Winner-Takes-All
            if self.cfg.pick_primary is True:
                winner_lbl = max(all_wspans, key=all_wspans.get)
                color = def_metal_col
                if winner_lbl != "metal":
                    if (w_idx := rec.get(winner_lbl, {}).get('idx')):
                        color = component_colors.get(winner_lbl, {}).get(w_idx, "black")
                abs_dE = abs(rec.get(winner_lbl, {}).get("dE", 0.0))
                (molecule_states_to_plot if winner_lbl != "metal" else metal_states_to_plot).append((abs_dE, E_full, color, hover_text, 'single'))
            
            # 2. Weighted Global Average
            elif self.cfg.pick_primary == "blended":
                final_rgb, total_ov_sum = np.zeros(3), 0.0
                for lbl in comp_iter_order:
                    for r in comps.get(lbl, []):
                        ov = r.get('ov', 0.0)
                        if ov > 1e-8:
                            c = component_colors.get(lbl, {}).get(int(r['comp_idx']), (0.4, 0.4, 0.4))
                            final_rgb += np.array(c[:3]) * ov
                            total_ov_sum += ov
                blend_col = tuple(final_rgb / total_ov_sum) if total_ov_sum > 0 else (0.4, 0.4, 0.4)
                abs_dE = abs(rec.get("metal", {}).get("dE", 0.0))
                (molecule_states_to_plot if total_mol_wspan >= self.cfg.min_total_mol_wspan else metal_states_to_plot).append((abs_dE, E_full, blend_col, hover_text, 'single'))

            # 3. Segmented / Stacked
            else:
                abs_dE = abs(rec.get("metal", {}).get("dE", 0.0))
                if total_mol_wspan >= self.cfg.min_total_mol_wspan:
                    mol_contribs = []
                    for mol_lbl in mol_labels:
                        if (m_recs := comps.get(mol_lbl)):
                            top = max(m_recs, key=lambda r: r.get("ov", 0.0))
                            if top.get("ov", 0.0) > 1e-6:
                                mol_contribs.append({'ov': top['ov'], 'col': component_colors[mol_lbl].get(int(top['comp_idx']))})
                    mol_contribs.sort(key=lambda x: x['ov'], reverse=True)
                    segments = []
                    if mol_contribs:
                        h = 1.0 / len(mol_contribs)
                        for i, contrib in enumerate(mol_contribs):
                            segments.append((1.0 - (i+1)*h, 1.0 - i*h, contrib['col']))
                    molecule_states_to_plot.append((abs_dE, E_full, segments, hover_text, 'multi'))
                else:
                    metal_states_to_plot.append((abs_dE, E_full, def_metal_col, hover_text, 'single'))

        # Final Sorted Rendering
        metal_states_to_plot.sort(key=lambda x: x[0]); molecule_states_to_plot.sort(key=lambda x: x[0])
        self._artists_f.clear(); self._hover_map_f.clear()
        for abs_dE, E_full, plot_data, hover_text, plot_type in metal_states_to_plot + molecule_states_to_plot:
            if plot_type == 'single':
                line = ax_f.vlines(E_full, 0, 1, color=plot_data, lw=self.cfg.lw_stick)
                self._artists_f.append(line); self._hover_map_f[line] = hover_text
            elif plot_type == 'multi':
                if not plot_data:
                    line = ax_f.vlines(E_full, 0, 1, color=def_metal_col, lw=self.cfg.lw_stick)
                    self._artists_f.append(line); self._hover_map_f[line] = hover_text
                else:
                    for y_bot, y_top, col in plot_data:
                        line = ax_f.vlines(E_full, y_bot, y_top, color=col, lw=self.cfg.lw_stick)
                        self._artists_f.append(line); self._hover_map_f[line] = hover_text

        ax_f.set_title(self.cfg.title_full); ax_f.set_ylabel(self.cfg.ylabel); ax_f.set_xlabel(self.cfg.xlabel)
        if self.cfg.energy_range: ax_f.set_xlim(self.cfg.energy_range)
        if self.cfg.annotate_on_hover and HAS_MPLCURSORS and self._artists_f:
            self._cursor_f = mplcursors.cursor(self._artists_f, hover=True)
            @self._cursor_f.connect("add")
            def _on_add_f(sel, hmap=self._hover_map_f):
                if (txt := hmap.get(sel.artist)): sel.annotation.set_text(txt)
        fig.tight_layout()
        return fig, axes