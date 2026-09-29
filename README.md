# Feature Engineering for Structure-Based Generative Learning

## Overview
This project implements a custom feature extraction and encoding-decoding pipeline for protein structures (PDB). We designed a **Residue-Level, Graph-Based Representation** of the protein and encoded it into structured numerical **PyTorch Tensors**. This format is specifically designed to serve as input to structure-based generative deep learning models.

## 1. Feature Choices & Justification
The pipeline utilizes a **Residue-Level (Coarse-Grained) Graph Representation**, which significantly reduces computational overhead compared to all-atom representations while preserving the true topological fold. The extracted features are strictly categorized to support robust generative modeling:

* **Biological Information:** 
  * *Features:* Amino acid identity (One-Hot Encoded), Chain IDs, and Sequence separation indices.
  * *Justification:* Provides the evolutionary and primary sequence context required for the model to understand the basic building blocks and polymer topology.
* **Chemical Information (Physicochemical):**
  * *Features:* Hydropathy, Charge, Volume, and Pharmacophore properties (H-Donors/Acceptors).
  * *Justification:* These are the primary driving forces behind protein folding and binding affinity. Feeding these explicitly allows generative models to condition on local chemistry, not just geometry.
* **Structural Information:**
  * *Features:* Alpha-Carbon (Cα) nodes and B-factors.
  * *Justification:* Cα nodes define the structural backbone trace. B-factors provide the model with a measure of local structural flexibility and experimental uncertainty.
* **Geometric Information:**
  * *Features:* 3D Coordinates (X,Y,Z), Phi/Psi dihedral angles, and side-chain orientation (CB) vectors.
  * *Justification:* Sine/Cosine encoded dihedral angles and CB vectors give the model perfect spatial orientation of how the backbone twists and where the side-chain is pointing.
* **Relationship Information (Graph Edges):**
  * *Features:* K-Nearest Neighbors (KNN) spatial edges (≤ 8.0 Å) with RBF-encoded distances and 3D directional unit vectors.
  * *Justification:* In generative drug discovery, non-sequential tertiary contacts define active sites. Explicitly filtering out sequential backbone bonds forces the model to learn long-range physical interactions rather than memorizing the 1D sequence.

## 2. Pipeline Data Flow (Inputs & Outputs)
To ensure strict modularity, the pipeline is divided into distinct ETL stages where each script has a rigorously defined input and output:

1. **Extractor (`pipeline/extractor.py`)**
   * **Input:** Raw `.pdb` structure file (e.g., `4hhb.pdb`).
   * **Output:** A list of Python dictionaries, where each dictionary holds the human-readable physical and chemical properties of a single amino acid.
2. **Encoder (`pipeline/encoder.py`)**
   * **Input:** The list of dictionaries from the extractor.
   * **Output:** Four standardized PyTorch Tensors ready for graph neural networks: Node features (`x`), coordinates (`pos`), connectivity (`edge_index`), and edge features (`edge_attr`).
3. **Decoder (`pipeline/decoder.py`)**
   * **Input:** The encoded PyTorch Tensors.
   * **Output:** A reconstructed list of dictionaries mapping the tensors back to human-readable biological data.
4. **Validator (`pipeline/validator.py`)**
   * **Input:** The original dictionary (from step 1) and the reconstructed dictionary (from step 3).
   * **Output:** A generated `validation_report.json` containing the structural RMSD error and Sequence Recovery Percentage.

## 3. Encoding and Decoding Logic
### Encoder Logic (`pipeline/encoder.py`)
The encoder is mathematically deterministic. It converts biological data into numerical tensors using the following logic:
* **Node Identity Encoding:** Amino acid types (e.g., "ALA") are mapped to a 21-dimensional **One-Hot Encoded vector** (20 standard amino acids + 1 'Unknown' token).
* **Continuous Feature Normalization:** Physicochemical properties (hydropathy, volume) are passed through as continuous float values.
* **Angular Encoding (Trigonometry):** To solve the boundary discontinuity problem of dihedral angles (where -180° and +180° are physically identical but mathematically distant), Phi and Psi angles are encoded as their `Sine` and `Cosine` components.
* **Edge Distance Encoding (RBF):** Instead of passing a single scalar distance (e.g., 5.2 Å), distances are expanded into a **16-dimensional Radial Basis Function (RBF)**. This smears the distance across 16 Gaussian bins, which allows a neural network to easily learn non-linear distance thresholds (e.g., recognizing strong vs. weak hydrogen bonds).
* **Directional Vectors:** The relative 3D unit vector `(x,y,z)` pointing from the source node to the target node is calculated to provide equivariant neural networks with spatial orientation.

### Decoder Logic (`pipeline/decoder.py`)
The decoder acts as a mathematical reverse-engineer to prove that no critical biological data was lost during tensor transformation.
* **Identity Decoding:** Applies `torch.argmax()` to the 21-dimensional one-hot array to retrieve the exact amino acid string (e.g., `[1, 0, 0...] -> 'ALA'`).
* **Angular Decoding:** Reconstructs the exact dihedral angles using the arctangent function: `np.arctan2(sin, cos)`.
* **Topology Reconstruction:** Decodes the `edge_index` tensor back into a source-target node list. It also dynamically recalculates sequential peptide bonds by checking if two nodes share the same Chain ID and have adjacent residue sequence numbers (`abs(res_i - res_j) == 1`).

## 4. Assumptions and Design Trade-offs
* **Trade-off (Coarse vs. All-Atom):** By using Cα representation, we assume side-chain positions can be inferred or reconstructed post-generation (e.g., via Rosetta or FastRelax). This trades atomic precision for massive ML training speedups.
* **Assumption (Missing Atoms):** The pipeline assumes residues missing Cα atoms are artifacts (e.g., poor experimental density) and skips them to maintain graph continuity.
* **Trade-off (Secondary Structure):** Explicit secondary structure classification was dropped in favor of raw Phi/Psi angles, assuming a powerful generative model will natively learn secondary structure representations from the angles and hydrogen bond potentials.

## 5. Scalability to Large Protein Datasets (Optional Bonus)
While the current pipeline exports human-readable JSON files for debugging and visualization, this is not scalable for training on millions of proteins (e.g., the PDB or AlphaFold DB).
To scale this pipeline:
1. **PyTorch DataLoaders:** The JSON export would be bypassed, directly yielding `torch_geometric.data.Data` objects.
2. **HDF5 / LMDB Storage:** Processed tensors would be chunked into Lightning Memory-Mapped Databases (LMDB) or HDF5 files to allow lightning-fast, parallelized GPU batching without the disk I/O bottleneck of parsing text files.
3. **Parallel Processing:** The `extractor.py` logic can be trivially wrapped in Python's `multiprocessing` pool, mapping the parser across thousands of `.pdb` files concurrently.

## 6. Usage
```bash
# Run the core ETL pipeline on the default PDB (7rfw.pdb)
# This will automatically generate the JSON, the Validation Report, AND the HTML Visualization in data/output/
python3 main.py

# Or specify any PDB file located in the data/input/ directory
python3 main.py 1crn.pdb
```
