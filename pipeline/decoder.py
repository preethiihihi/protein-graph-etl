import torch
import numpy as np

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_AA = 'UNK'


class FeatureDecoder:
    def __init__(self):
        pass

    def decode(self, encoded_data):
        # Support both PyG Data objects and raw dictionaries
        is_pyg = not isinstance(encoded_data, dict) and hasattr(encoded_data, 'x')
        
        x = encoded_data.x if is_pyg else encoded_data['x']
        pos = encoded_data.pos if is_pyg else encoded_data['pos']
        
        reconstructed = []
        num_nodes = x.shape[0]
        
        for i in range(num_nodes):
            node_feat = x[i]
            
            # 1. Identity (first 21 dims)
            aa_idx = torch.argmax(node_feat[:21]).item()
            resname = IDX_TO_AA.get(aa_idx, UNKNOWN_AA)
            
            # 2. Reconstruct angles from sin/cos (dims 26-29)
            sin_phi, cos_phi = node_feat[26].item(), node_feat[27].item()
            sin_psi, cos_psi = node_feat[28].item(), node_feat[29].item()
            
            phi = np.arctan2(sin_phi, cos_phi)
            psi = np.arctan2(sin_psi, cos_psi)
            
            if is_pyg:
                meta = getattr(encoded_data, 'node_metadata', [])
            else:
                meta = encoded_data.get('node_metadata', [])
            if meta:
                chain_id = meta[i]['chain_id']
                resseq = meta[i]['residue_number']
            else:
                chain_id = 'A'
                resseq = i + 1
            
            reconstructed.append({
                'resname': resname,
                'chain': chain_id,
                'residue_number': resseq,
                'phi_reconstructed': phi,
                'psi_reconstructed': psi
            })
            
        coords = pos.cpu().numpy()
        
        # Decode the side-chain orientation vectors
        if is_pyg:
            cb_vec = getattr(encoded_data, 'cb_vec', None)
            edge_index = getattr(encoded_data, 'edge_index', None)
            edge_types = getattr(encoded_data, 'edge_types', [])
        else:
            cb_vec = encoded_data.get('cb_vec')
            edge_index = encoded_data.get('edge_index')
            edge_types = encoded_data.get('edge_types', [])
            
        cb_coords = cb_vec.cpu().numpy() if cb_vec is not None else None
        spatial_edges = []
        
        if edge_index is not None and edge_index.shape[1] > 0:
            for k in range(edge_index.shape[1]):
                src = edge_index[0, k].item()
                dst = edge_index[1, k].item()
                e_type = edge_types[k] if k < len(edge_types) else 'SPATIAL'
                spatial_edges.append({
                    'source': src,
                    'target': dst,
                    'type': e_type
                })
                
        # Infer Sequential Edges from Metadata
        sequential_edges = []
        for i in range(num_nodes - 1):
            for j in range(i + 1, num_nodes):
                chain_i = reconstructed[i]['chain']
                chain_j = reconstructed[j]['chain']
                resseq_i = reconstructed[i]['residue_number']
                resseq_j = reconstructed[j]['residue_number']
                
                if chain_i == chain_j and abs(resseq_i - resseq_j) == 1:
                    # found a sequential connection
                    sequential_edges.append({
                        'source': i,
                        'target': j,
                        'type': 'SEQUENTIAL'
                    })
        
        return {
            'residues_meta': reconstructed,
            'coords': coords,
            'cb_vec': cb_coords,
            'spatial_edges': spatial_edges,
            'sequential_edges': sequential_edges
        }
