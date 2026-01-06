import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Locpot
from numpy import array, shape, dot
from os.path import exists, getsize
import sys

# --- Helper Functions (Shared) ---

def parse_doscar(filepath):
    with open(filepath,'r') as file:
        line=file.readline().split()
        atomnum=int(line[0])
        for i in range(5):
            line=file.readline().split()
        nedos=int(line[2])
        ef=float(line[3])
        dos=[]
        energies=[]
        for i in range(atomnum+1):
            if i!=0:
                line=file.readline()
            for j in range(nedos):
                line=file.readline().split()
                if i==0:
                    energies.append(float(line[0]))
                if j==0:
                    temp_dos=[[] for k in range(len(line)-1)]
                for k in range(len(line)-1):
                    temp_dos[k].append(float(line[k+1]))
            dos.append(temp_dos)
    energies=array(energies)-ef
    
    num_columns=shape(dos[1:])[1]
    if num_columns==3:
        orbitals=['s','p','d']
    elif num_columns==6:
        orbitals=['s_up','s_down','p_up','p_down','d_up','d_down']
    elif num_columns==9:
        orbitals=['s','p_y','p_z','p_x','d_xy','d_yz','d_z2','d_xz','d_x2-y2']
    elif num_columns==18:
        orbitals=['s_up','s_down','p_y_up','p_y_down','p_z_up','p_z_down','p_x_up','p_x_down','d_xy_up','d_xy_down','d_yz_up','d_yz_down','d_z2_up','d_z2_down','d_xz_up','d_xz_down','d_x2-y2_up','d_x2-y2_down']
        
    return dos, energies, ef, orbitals

def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start=8
            mode=lines[7].split()[0]
        else:
            mode=lines[8].split()[0]
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        if mode!='Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j]>1.0 or coord[i][j]<0.0:
                        if coord[i][j]>1.0:
                            coord[i][j]-=1.0
                        elif coord[i][j]<0.0:
                            coord[i][j]+=1.0
                coord[i]=dot(coord[i],latticevectors)
            
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums
    
def load_locpot(locpot_path):
    return Locpot.from_file(locpot_path)

def parse_VASP_output(self, **args):
    if 'doscar' in args:
        doscar = args['doscar']
    else:
        doscar = './DOSCAR'
        
    if 'poscar' in args:
        poscar = args['poscar']
    elif exists('./CONTCAR'):
        if getsize('./CONTCAR') > 0:
            poscar = './CONTCAR'
        else:
            poscar = './POSCAR'
    else:
        poscar = './POSCAR'
            
    try:
        self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar)[:4]
        self.dos, self.energies, self.ef, self.orbitals = parse_doscar(doscar)
    except Exception as e:
        print('Error reading input files:', e)
        sys.exit()

# --- Main Calculation Function ---

def calculate_work_function(locpot, ef, curvature_tol=5e-8, dipole_threshold=0.15, min_width=10, plot=True, verbose=True):
    """
    Calculates work function using robust wraparound handling.
    plot: Boolean to toggle matplotlib output.
    verbose: Boolean to toggle console printing.
    """
    # Extract and average the potential
    potential_data = locpot.data['total']
    z_potential = np.mean(np.mean(potential_data, axis=0), axis=0)
    nz = z_potential.shape[0]
    z_coords = np.arange(nz)
    
    # --- 1. Periodic Curvature Calculation ---
    z_padded = np.pad(z_potential, (2, 2), mode='wrap')
    slopes_padded = np.gradient(z_padded)
    curvatures_padded = np.abs(np.gradient(slopes_padded))
    curvatures = curvatures_padded[2:-2]
    
    # --- 2. Identify Linear Regions (Raw) ---
    stable_indices = np.where(curvatures < curvature_tol)[0]
    
    if stable_indices.size == 0:
        if verbose: print(f"[ERROR] No stable curvature region found (tol={curvature_tol}).")
        return ef, np.max(z_potential), np.max(z_potential) - ef

    # Group contiguous stable indices
    linear_regions = []
    if stable_indices.size > 0:
        start = stable_indices[0]
        for i in range(1, len(stable_indices)):
            if stable_indices[i] != stable_indices[i - 1] + 1:
                end = stable_indices[i - 1] + 1
                linear_regions.append({
                    'start': start, 'end': end, 
                    'width': end - start, 
                    'avg_v': np.mean(z_potential[start:end]),
                    'is_wrapped': False
                })
                start = stable_indices[i]
        end = stable_indices[-1] + 1
        linear_regions.append({
            'start': start, 'end': end, 
            'width': end - start, 
            'avg_v': np.mean(z_potential[start:end]),
            'is_wrapped': False
        })

    # --- 3. ROBUST WRAPAROUND STITCHING ---
    regions = []
    if len(linear_regions) > 1:
        first_r = linear_regions[0]
        last_r = linear_regions[-1]
        
        # Check if the last region touches the end (N) and first touches start (0)
        touches_start = (first_r['start'] <= 1)
        touches_end = (last_r['end'] >= nz - 1)
        
        if touches_start and touches_end:
            # Merge them into one "Wrapped" region
            total_points = first_r['width'] + last_r['width']
            weighted_v = (first_r['avg_v'] * first_r['width'] + last_r['avg_v'] * last_r['width']) / total_points
            
            wrapped_region = {
                'start': last_r['start'], 
                'end': first_r['end'],    
                'width': total_points, 
                'avg_v': weighted_v,
                'is_wrapped': True,
                'segments': [last_r, first_r] 
            }
            regions = [wrapped_region] + linear_regions[1:-1]
        else:
            regions = linear_regions
    else:
        regions = linear_regions

    # --- 4. Filtering and Selection ---
    
    # A. Filter by Potential Height (must be "vacuum-like")
    v_min = np.min(z_potential)
    v_max = np.max(z_potential)
    midpoint = (v_min + v_max) / 2
    
    candidates = []
    for r in regions:
        if r['avg_v'] > midpoint and r['width'] >= min_width:
            candidates.append(r)
            
    if not candidates:
        if verbose: print("[WARN] No valid vacuum candidates found after filtering. Reverting to widest raw region.")
        candidates = regions

    # B. Sort Spatially
    for r in candidates:
        if r['is_wrapped']:
             r['sort_index'] = nz 
        else:
             r['sort_index'] = r['start']
             
    candidates.sort(key=lambda x: x['sort_index'])

    # C. Dipole Detection
    best_region = None
    selection_mode = "Default"
    dipole_found = False
    
    if len(candidates) > 1:
        max_step_diff = 0
        split_index = -1 
        
        for i in range(len(candidates) - 1):
            diff = abs(candidates[i]['avg_v'] - candidates[i+1]['avg_v'])
            if diff > max_step_diff:
                max_step_diff = diff
                split_index = i
        
        if max_step_diff > dipole_threshold:
            dipole_found = True
            post_step_regions = candidates[split_index+1:]
            if not post_step_regions:
                post_step_regions = [candidates[-1]]
            best_region = max(post_step_regions, key=lambda x: x['width'])
            selection_mode = "Post-Step (Right/Wrapped)"
        else:
            best_region = max(candidates, key=lambda x: x['width'])
            selection_mode = "Widest (No Step)"
    else:
        best_region = candidates[0]
        selection_mode = "Single Candidate"

    # --- 5. Calculate Final Values ---
    if best_region['is_wrapped']:
        seg1 = best_region['segments'][0] 
        seg2 = best_region['segments'][1] 
        
        chunk1 = potential_data[:, :, seg1['start']:seg1['end']]
        chunk2 = potential_data[:, :, seg2['start']:seg2['end']]
        
        vacuum_potential = (np.sum(chunk1) + np.sum(chunk2)) / (chunk1.size + chunk2.size)
        v_start_label = f"Wrapped ({seg1['start']}-{seg1['end']} & {seg2['start']}-{seg2['end']})"
    else:
        v_start = best_region['start']
        v_end = best_region['end']
        vacuum_potential = np.mean(potential_data[:, :, v_start:v_end])
        v_start_label = f"{v_start}-{v_end}"

    work_function = vacuum_potential - ef

    # --- Plotting ---
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 1. Potential Plot
        ax1.plot(z_coords, z_potential, 'b-', label='Planar Averaged Potential')
        ax1.axhline(ef, color='k', linestyle='--', label=f'Fermi Energy ({ef:.2f} eV)')

        # Draw Selection
        if best_region['is_wrapped']:
            s1, s2 = best_region['segments']
            ax1.axvspan(s1['start'], s1['end'], color='green', alpha=0.3, label='Selected Vacuum')
            ax1.axvspan(s2['start'], s2['end'], color='green', alpha=0.3)
        else:
            ax1.axvspan(best_region['start'], best_region['end'], color='green', alpha=0.3, label='Selected Vacuum')

        # Highlight rejected candidates (Grey)
        for r in candidates:
            if r != best_region:
                if r['is_wrapped']:
                    s1, s2 = r['segments']
                    ax1.axvspan(s1['start'], s1['end'], color='gray', alpha=0.1)
                    ax1.axvspan(s2['start'], s2['end'], color='gray', alpha=0.1)
                else:
                    ax1.axvspan(r['start'], r['end'], color='gray', alpha=0.1)

        ax1.set_ylabel("Potential (eV)")
        ax1.set_title(f"Vac: {vacuum_potential:.4f} eV | Ef: {ef:.4f} eV | WF: {work_function:.4f} eV\nMode: {selection_mode}")
        ax1.legend(loc='lower left')
        
        # 2. Curvature Plot
        ax2.plot(z_coords, curvatures, 'r-', label='Curvature |d²V/dz²|')
        ax2.axhline(curvature_tol, color='k', linestyle='--', label='Tolerance')
        ax2.set_ylabel("Change in Slope")
        ax2.set_xlabel("Grid Points (z)")
        ax2.set_yscale('log')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # --- Console Output ---
    if verbose:
        print(f"Dipole Found: {'Yes' if dipole_found else 'No'}")
        print(f"Selection Mode: {selection_mode}")
        print(f"Selected Region: {v_start_label}")
        print(f"Vacuum Potential: {vacuum_potential:.4f} eV")
        print(f"Fermi Energy: {ef:.4f} eV")
        print(f"Work Function: {work_function:.4f} eV")
    
    return ef, vacuum_potential, work_function

# --- Execution ---
if __name__ == "__main__":
    locpot_path = 'LOCPOT' # Update path as needed
    doscar_path = 'DOSCAR' # Update path as needed
    
    if exists(locpot_path):
        locpot = load_locpot(locpot_path)
        try:
            dos, energies, ef, orbitals = parse_doscar(doscar_path)
        except Exception as e:
            print(f"Warning: Could not parse DOSCAR ({e}). Using default EF=0.0")
            ef = 0.0
            
        calculate_work_function(locpot, ef, dipole_threshold=0.15, min_width=10, plot=True, verbose=True)
    else:
        print(f"File not found: {locpot_path}")