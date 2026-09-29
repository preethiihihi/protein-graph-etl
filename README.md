# 🧬 Structure-Based Generative Learning: Feature Pipeline

A rigorous biocomputing pipeline designed to extract, transform, and encode raw Protein Data Bank (PDB) structures into **residue-level, graph-based PyTorch Tensors**—the exact machine-readable format required for structure-based generative deep learning models (SBDD).

---

## 🚀 Quick Start & How to Use

1. **Set up a Python Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. **Install the required biophysics and ML libraries:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the full pipeline on the default sample:**
   ```bash
   python3 main.py
   ```
3. **Run on your own specific PDB file:**
   Simply place your `.pdb` file inside the `data/input/` folder and pass it as an argument:
   ```bash
   python3 main.py your_custom_protein.pdb
   ```

**Where are the results?**
All generated data will instantly appear in the **`data/output/`** folder! This includes:
* `protein_graph.pt`: The pure PyTorch Geometric tensor object.
* `full_protein_graph.json`: A human-readable export of the graph.
* `validation_report.json`: Proof of structural integrity (RMSD metrics).
* `protein_3d_graph.html`: An interactive 3D visualization you can open in your browser!

---

## Brief Repository Structure

* **`data/input/`** - Drop your raw `.pdb` structure files here.
* **`data/output/`** - Contains all generated AI tensors, JSON graphs, validation reports, and 3D HTML visualizations.
* **`pipeline/extractor.py`** - Parses physical, geometric, and chemical features from the PDB.
* **`pipeline/encoder.py`** - Converts the biological features into a mathematical K-NN graph of PyTorch Geometric tensors.
* **`pipeline/decoder.py`** - Reverse-engineers the tensors back into readable biology.
* **`pipeline/validator.py`** - Scores the pipeline's fidelity (Structural RMSD & Sequence Recovery).
* **`main.py`** - The core execution script that ties the entire ETL pipeline together.

---

## Logic Overview

This pipeline acts as the essential "translator" between raw biology and an Artificial Intelligence model. 

1. **Extraction (Biology → Data):** It parses the protein and extracts coordinates, angles, and physicochemical traits at the **residue-level (Cα)**. This trades all-atom precision for massive computational speedups during ML training.
2. **Graph Construction (Nodes & Edges):** Amino acids become graph nodes (embedded with 31-dimensional chemical and geometric features). Spatial neighbors (≤ 8.0 Å) become graph edges. *Crucially, sequential peptide bonds are deliberately dropped to force the generative AI to learn long-range 3D folding rules rather than just memorizing a 1D sequence.*
3. **Encoding (Data → Tensors):** Distances are expanded using 16-bin Radial Basis Functions (RBFs), angles are trigonometrically encoded (Sine/Cosine to prevent boundary errors), and categorical traits are One-Hot encoded. Everything is packaged into a strict PyTorch Geometric `Data` object.
4. **Validation (Decoding):** The tensors are decoded back into a physical structure to calculate the mathematical RMSD, proving that absolutely zero structural data was lost during the AI embedding process.

*(For an extremely deep dive into the biophysics, tensor dimensionalities, and scalability trade-offs, please refer to the detailed `PDB_Pipeline_Documentation.md` file provided alongside this repository!)*
