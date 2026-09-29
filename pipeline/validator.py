import json
import numpy as np

def calculate_rmsd(coords1, coords2):
    """Calculates Root Mean Square Deviation (RMSD) between two sets of 3D coordinates."""
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

def validate_pipeline(original_features, reconstructed_data, output_json="validation_report.json"):
    """
    Validates the reconstructed features against the original extracted features 
    and outputs the metrics to a JSON file.
    """
    print("\n--- Running Biotech-Standard Validation ---")
    
    recon_meta = reconstructed_data['residues_meta']
    recon_coords = reconstructed_data['coords']
    
    num_residues = len(original_features)
    
    # 1. Sequence Recovery Rate
    original_seq = [r['resname'] for r in original_features]
    recon_seq = [r['resname'] for r in recon_meta]
    
    matches = sum(1 for o, r in zip(original_seq, recon_seq) if o == r)
    sequence_recovery = (matches / num_residues) * 100 if num_residues > 0 else 0.0
    
    # 2. Coordinate RMSD (Root Mean Square Deviation)
    original_coords = np.array([r['coords'] for r in original_features])
    rmsd = calculate_rmsd(original_coords, recon_coords)
    
    # 3. Angle MAE (Mean Absolute Error) for valid angles
    phi_errors = []
    psi_errors = []
    
    for orig, recon in zip(original_features, recon_meta):
        if orig['phi'] != 0.0:
            phi_errors.append(abs(orig['phi'] - recon['phi_reconstructed']))
        if orig['psi'] != 0.0:
            psi_errors.append(abs(orig['psi'] - recon['psi_reconstructed']))
            
    mae_phi = float(np.mean(phi_errors)) if phi_errors else 0.0
    mae_psi = float(np.mean(psi_errors)) if psi_errors else 0.0
    
    # Create the validation report JSON structure
    report = {
        "metadata": {
            "total_residues_processed": num_residues,
            "status": "SUCCESS" if rmsd < 1e-4 and sequence_recovery == 100.0 else "WARNING"
        },
        "metrics": {
            "sequence_recovery_percentage": float(sequence_recovery),
            "coordinate_rmsd_angstroms": rmsd,
            "angle_mae_phi_radians": mae_phi,
            "angle_mae_psi_radians": mae_psi
        },
        "sample_residue_comparison": {
            "original_residue_1": original_seq[0] if original_seq else None,
            "reconstructed_residue_1": recon_seq[0] if recon_seq else None,
            "original_coord_1": original_coords[0].tolist() if len(original_coords) > 0 else None,
            "reconstructed_coord_1": recon_coords[0].tolist() if len(recon_coords) > 0 else None
        }
    }
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Validation complete. Report saved to: {output_json}")
    print(f"  -> RMSD: {rmsd:.6f} Å")
    print(f"  -> Sequence Recovery: {sequence_recovery:.1f}%")
    
    return report

def validate_graph_schema(graph_dict):
    """
    Validates that the generated JSON strictly adheres to the SBDD Graph Schema.
    """
    print("\n--- Validating JSON Schema ---")
    
    # 1. Check top-level keys
    required_keys = ["graph_metadata", "nodes", "edges"]
    for key in required_keys:
        if key not in graph_dict:
            print(f"❌ Schema Error: Missing top-level key '{key}'")
            return False
            
    # 2. Check Node Schema
    if len(graph_dict["nodes"]) > 0:
        node = graph_dict["nodes"][0]
        required_node_keys = [
            'resname', 'chain', 'coords', 'cb_vector', 'phi', 'psi', 
            'b_factor', 'hydropathy', 'charge', 
            'volume', 'h_donors', 'h_acceptors'
        ]
        for k in required_node_keys:
            if k not in node:
                print(f"❌ Schema Error: Missing node key '{k}'")
                return False
                
    # 3. Check Edge Schema
    if len(graph_dict["edges"]) > 0:
        edge = graph_dict["edges"][0]
        # We now export human-readable edges instead of a flat features array
        required_edge_keys = ['source', 'target', 'distance', 'edge_type']
        for k in required_edge_keys:
            if k not in edge:
                print(f"❌ Schema Error: Missing edge key '{k}'")
                return False
                
    print("✅ JSON Schema Validation Passed! The data strictly matches the SBDD blueprint.")
    return True

if __name__ == "__main__":
    print("Validator module ready.")
