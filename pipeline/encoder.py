import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_IDX = len(AMINO_ACIDS)

SS_TO_IDX = {'HELIX': 0, 'SHEET': 1, 'COIL': 2}

class MockData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class FeatureEncoder:
    def __init__(self, k_neighbors=10, distance_threshold=8.0):
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

    def _rbf(self, D, D_min=0.0, D_max=20.0, D_count=16):
        """Radial Basis Function encoding for distances."""
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma)**2)

    def encode(self, residues_data):
        num_nodes = len(residues_data)
        
        # Initialize Node Feature Tensors
        # Initialize Node Feature Tensors
        # One-hot(21) + Phys(3) + HBonds(2) + Angles(4) + B-factor(1) = 31 dimensions
        x = torch.zeros((num_nodes, 31), dtype=torch.float32)
        pos = torch.zeros((num_nodes, 3), dtype=torch.float32)
        cb_vec = torch.zeros((num_nodes, 3), dtype=torch.float32)
        
        # 1. Store node metadata to identify residues
        node_metadata = []
        
        for i, res in enumerate(residues_data):
            # Capture sequence info for each node
            chain_id = res.get('chain', 'A')
            resseq = res.get('residue_number', res.get('resseq', i + 1))
            resname = res.get('resname', 'UNK')
            
            node_metadata.append({
                'node_id': int(i),
                'chain_id': str(chain_id),
                'residue_number': int(resseq),
                'resname': str(resname),
                'sequence_index': int(i)
            })
            
            # 1. Identity (0-20)
            idx = AA_TO_IDX.get(resname, UNKNOWN_IDX)
            x[i, idx] = 1.0
            
            # 2. Physicochemical (21-23)
            x[i, 21] = res['charge']
            x[i, 22] = res['hydropathy']
            x[i, 23] = res['volume'] / 200.0  # rough normalization
            
            # 3. Pharmacophore H-bonds (24-25)
            x[i, 24] = res['h_donors']
            x[i, 25] = res['h_acceptors']
            
            # 4. Angles (26-29): sin/cos of phi and psi
            x[i, 26] = np.sin(res['phi'])
            x[i, 27] = np.cos(res['phi'])
            x[i, 28] = np.sin(res['psi'])
            x[i, 29] = np.cos(res['psi'])
            
            # 5. Metadata (30)
            x[i, 30] = res['b_factor'] / 100.0 # rough normalization
            
            # Separate Vector Tensors
            pos[i] = torch.tensor(res['coords'], dtype=torch.float32)
            cb_vec[i] = torch.tensor(res['cb_vector'], dtype=torch.float32)

        # Edges Construction
        coords = pos.numpy()
        if num_nodes == 0:
            return {'x': x, 'pos': pos, 'edge_index': torch.empty((2, 0)), 'edge_attr': torch.empty((0, 16))}
            
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, num_nodes), algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        src_nodes, dst_nodes, edge_dists, seq_seps = [], [], [], []
        direction_vectors = []
        same_chain_flags = [] # Replaces peptide_bonds flag
        edge_types = []       # INTRA_CHAIN_SPATIAL or INTER_CHAIN_SPATIAL
        salt_bridges = []
        
        for i in range(num_nodes):
            for j, dist in zip(indices[i], distances[i]):
                if i != j and dist <= self.distance_threshold:
                    meta_i = node_metadata[i]
                    meta_j = node_metadata[j]
                    
                    chain_i = meta_i['chain_id']
                    chain_j = meta_j['chain_id']
                    resseq_i = meta_i['residue_number']
                    resseq_j = meta_j['residue_number']
                    
                    same_chain = (chain_i == chain_j)
                    # 2. Check sequential logic based on PDB residue numbers
                    is_sequential = same_chain and abs(resseq_i - resseq_j) == 1
                    
                    # 3. Do NOT create sequential edges
                    if is_sequential:
                        continue
                        
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_dists.append(dist)
                    
                    # Sequence separation
                    if same_chain:
                        seq_sep = max(min(resseq_j - resseq_i, 5), -5) 
                    else:
                        seq_sep = 0 # Not biologically meaningful for inter-chain
                    seq_seps.append(seq_sep)
                    
                    # 4. Same chain flag and Edge Type
                    # The previous 'is_peptide' flag is now redundant because we drop sequential edges.
                    # We reuse its tensor dimension for 'same_chain' to preserve the 32-dim edge_attr size.
                    same_chain_val = 1.0 if same_chain else 0.0
                    same_chain_flags.append([same_chain_val])
                    
                    edge_type = 'INTRA_CHAIN_SPATIAL' if same_chain else 'INTER_CHAIN_SPATIAL'
                    edge_types.append(edge_type)
                    
                    # Relative Space Direction Vector (Normalized)
                    vec = coords[j] - coords[i]
                    vec_normalized = vec / (dist + 1e-6)
                    direction_vectors.append(vec_normalized)
                    
                    # Proxy Salt Bridge Flag (Opposite charges & Distance < 5.0A)
                    charge_i = x[i, 21].item()
                    charge_j = x[j, 21].item()
                    is_salt_bridge = 1.0 if (charge_i * charge_j < 0 and dist < 5.0) else 0.0
                    salt_bridges.append([is_salt_bridge])
                    
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        # Encode continuous/categorical features
        rbf_features = self._rbf(torch.tensor(edge_dists, dtype=torch.float32))
        
        seq_seps_shifted = torch.tensor(seq_seps, dtype=torch.long) + 5
        seq_sep_onehot = torch.nn.functional.one_hot(seq_seps_shifted, num_classes=11).float()
        
        same_chain_tensor = torch.tensor(same_chain_flags, dtype=torch.float32)
        salt_bridge_tensor = torch.tensor(salt_bridges, dtype=torch.float32)
        dir_vec_tensor = torch.tensor(np.array(direction_vectors), dtype=torch.float32)
        
        # Combine all scalar and vector edge features (16 + 11 + 1 + 1 + 3 = 32 dims)
        # Note: same_chain_tensor replaces pep_bond_tensor
        edge_attr = torch.cat([rbf_features, seq_sep_onehot, same_chain_tensor, salt_bridge_tensor, dir_vec_tensor], dim=1)
        
        # 5. Export Human-Readable Edges for JSON
        raw_edges = []
        for k in range(len(src_nodes)):
            raw_edges.append({
                "source": getattr(src_nodes[k], "item", lambda: int(src_nodes[k]))(),
                "target": getattr(dst_nodes[k], "item", lambda: int(dst_nodes[k]))(),
                "edge_type": str(edge_types[k]),
                "distance": round(float(edge_dists[k]), 4),
                "sequence_separation": int(seq_seps[k]),
                "is_same_chain": bool(same_chain_flags[k][0]),
                "is_salt_bridge": bool(salt_bridges[k][0]),
                "direction_vector": [round(float(v), 4) for v in direction_vectors[k].tolist()]
            })
        
        try:
            from torch_geometric.data import Data
        except ImportError:
            Data = MockData
                        
        pyg_data = Data(
            x=x,
            pos=pos,
            cb_vec=cb_vec,
            edge_index=edge_index,
            edge_attr=edge_attr, # shape [E, 32]
            node_metadata=node_metadata,
            edge_types=edge_types,
            raw_edges=raw_edges
        )
        
        return pyg_data
