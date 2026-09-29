import numpy as np
import warnings
from Bio.PDB import PDBParser, PPBuilder
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

# Physicochemical mappings
HYDROPATHY = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 
              'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 
              'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 
              'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
CHARGE = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0}
VOLUME = {'GLY': 60.1, 'ALA': 88.6, 'SER': 89.0, 'CYS': 108.5, 'ASP': 111.1, 
          'THR': 116.1, 'ASN': 117.7, 'PRO': 112.7, 'VAL': 140.0, 'GLU': 138.4, 
          'GLN': 143.9, 'HIS': 153.2, 'MET': 162.9, 'ILE': 166.7, 'LEU': 166.7, 
          'LYS': 168.6, 'ARG': 173.4, 'PHE': 189.9, 'TYR': 193.6, 'TRP': 227.8}
H_DONORS = {'ARG': 5, 'LYS': 3, 'ASN': 2, 'GLN': 2, 'TRP': 1, 'HIS': 2, 'SER': 1, 'THR': 1, 'TYR': 1, 'CYS': 1}
H_ACCEPTORS = {'ASP': 4, 'GLU': 4, 'ASN': 2, 'GLN': 2, 'HIS': 1, 'SER': 2, 'THR': 2, 'TYR': 1, 'MET': 1}

def extract_features_from_pdb(pdb_file_path):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file_path)
    
    ppb = PPBuilder()
    
    residues_data = []
    
    # Iterate through peptides to easily get valid backbone phi/psi angles
    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            for poly in peptides:
                phi_psi_list = poly.get_phi_psi_list()
                
                for (res, (phi, psi)) in zip(poly, phi_psi_list):
                    resname = res.get_resname()
                    
                    if not res.has_id('CA'):
                        continue
                        
                    ca_atom = res['CA']
                    
                    # 1. Identity & Geometry
                    coords = ca_atom.get_coord().tolist()  # Convert to list for JSON serialization
                    b_factor = float(ca_atom.get_bfactor()) # Ensure float
                    
                    # 2. Side-Chain Proxy Frame (CB Vector)
                    cb_vector = [0.0, 0.0, 0.0]
                    if res.has_id('CB'):
                        cb_coords = res['CB'].get_coord()
                        ca_coords = ca_atom.get_coord()
                        vec = cb_coords - ca_coords
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            cb_vector = (vec / norm).tolist()
                    
                    # Store features
                    residues_data.append({
                        'resname': resname,
                        'chain': str(chain.get_id()),
                        'residue_number': int(res.get_id()[1]),
                        'coords': coords,
                        'cb_vector': cb_vector,
                        'phi': float(phi) if phi is not None else 0.0,
                        'psi': float(psi) if psi is not None else 0.0,
                        'b_factor': b_factor,
                        'hydropathy': HYDROPATHY.get(resname, 0.0),
                        'charge': CHARGE.get(resname, 0.0),
                        'volume': VOLUME.get(resname, 150.0),
                        'h_donors': H_DONORS.get(resname, 0),
                        'h_acceptors': H_ACCEPTORS.get(resname, 0)
                    })
    return residues_data

if __name__ == "__main__":
    print("Extractor module updated with rich features.")
