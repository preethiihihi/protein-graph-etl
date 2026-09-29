import os
import sys
import json
import numpy as np
from pipeline.extractor import extract_features_from_pdb
from pipeline.encoder import FeatureEncoder
from pipeline.decoder import FeatureDecoder

def download_sample_pdb(filename="1crn.pdb"):
    if not os.path.exists(filename):
        print(f"Downloading sample PDB {filename}...")
        import urllib.request
        url = "https://files.rcsb.org/download/1CRN.pdb"
        urllib.request.urlretrieve(url, filename)
    return filename

def main():
    print("--- Structure-Based Generative Learning Pipeline ---")
    
    # Clean up old output files to prevent reading stale data
    output_dir = os.path.join("data", "output")
    for old_file in ["full_protein_graph.json", "validation_report.json", "protein_3d_graph.html"]:
        old_path = os.path.join(output_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)
    
    # Check if user provided a filename argument
    filename = "7rfw.pdb"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    pdb_path = os.path.join("data", "input", filename)
    
    if not os.path.exists(pdb_path):
        print(f"Error: File not found at {pdb_path}")
        print("Please ensure the PDB file is placed in the 'data/input/' directory.")
        return

    print(f"\n[1] Parsing PDB File & Extracting Rich Features: {pdb_path}")
    
    try:
        extracted_features = extract_features_from_pdb(pdb_path)
        print(f"    Extracted {len(extracted_features)} valid peptide residues.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        return

    print("\n[2] Encoding to Machine-Learning Format (Graph)...")
    encoder = FeatureEncoder(k_neighbors=10, distance_threshold=8.0)
    encoded_graph = encoder.encode(extracted_features)
    
    print(f"\n--- EXPLORING THE PyG DATA OBJECT ---")
    
    # 1. Save the actual PyTorch Geometric Data object to disk
    import torch
    pyg_save_path = "data/output/protein_graph.pt"
    torch.save(encoded_graph, pyg_save_path)
    print(f"    [+] Saved pure PyTorch Tensors to '{pyg_save_path}'")
    
    # 2. Save the human-readable exploration to a text file
    exploration_text = f"""--- EXPLORING THE PyG DATA OBJECT ---
Summary: {encoded_graph}

1. Let's look at the first 3 Node Coordinates (pos):
{getattr(encoded_graph, 'pos')[:3]}

2. Let's look at the Edge Connections (edge_index) - first 5 edges:
{getattr(encoded_graph, 'edge_index')[:, :5]}

3. Let's look at Node #0's raw features (x):
{getattr(encoded_graph, 'x')[0]}
-------------------------------------
"""
    with open("data/output/pyg_exploration.txt", "w") as f:
        f.write(exploration_text)
    
    print(exploration_text)
    print(f"    Node Feature Tensor (X) Shape: {getattr(encoded_graph, 'x').shape} (34 dimensions!)")
    print(f"    Coordinate Tensor (pos) Shape: {getattr(encoded_graph, 'pos').shape}")
    print(f"    Edge Index Shape: {getattr(encoded_graph, 'edge_index').shape}")
    print(f"    Edge Features Shape: {getattr(encoded_graph, 'edge_attr').shape}")
    

    # --- NEW: Export the Unified Graph to JSON ---
    edges_list = getattr(encoded_graph, 'raw_edges', [])
        
    # Build the unified Node-Link format
    full_graph = {
        "graph_metadata": {
            "source_file": pdb_path,
            "num_nodes": len(extracted_features),
            "num_edges": len(edges_list)
        },
        "nodes": extracted_features,
        "edges": edges_list
    }
    
    unified_json_path = "data/output/full_protein_graph.json"
    with open(unified_json_path, "w") as f:
        json.dump(full_graph, f, indent=4)
    print(f"    [+] Successfully saved the unified Node-Link graph to '{unified_json_path}'")
    
    # --- Schema Validation ---
    from pipeline.validator import validate_graph_schema
    validate_graph_schema(full_graph)
    # ---------------------------------------------
    
    print("\n[3] Decoding back to structural representation (from PyG Data)...")
    decoder = FeatureDecoder()
    
    # We now pass the PyG `Data` object directly into the decoder instead of the raw dictionary!
    reconstructed = decoder.decode(encoded_graph)
    recon_meta = reconstructed['residues_meta']
    
    print(f"    Reconstructed {len(recon_meta)} residues.")
    
    from pipeline.validator import validate_pipeline
    report_path = "data/output/validation_report.json"
    report = validate_pipeline(extracted_features, reconstructed, output_json=report_path)
    
    print(f"\nPipeline executed successfully! Check {report_path} for the final structural assessment.")
    
    # --- Auto-Generate Visualization ---
    print("\n[4] Generating 3D Interactive Visualization...")
    from visualizations.visualize_3d_graph import generate_3d_visualization
    html_out = "data/output/protein_3d_graph.html"
    generate_3d_visualization(unified_json_path, html_out)

if __name__ == "__main__":
    main()
