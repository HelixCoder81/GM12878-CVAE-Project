GM12878-CVAE-Pro: 3D Chromatin Structure ReconstructionThis repository provides a high-performance implementation of a Conditional Variational Autoencoder (CVAE) designed for 3D chromatin structure reconstruction. By integrating Hi-C contact matrices with multi-dimensional epigenetic features (TADs, A/B Compartments, and Loops), the model generates high-resolution 3D trajectories from 2D genomic interaction data.🌟 Key FeaturesResidual CVAE Architecture: Utilizes a Deep Residual Convolutional Encoder (ResNet-style) to extract intricate spatial patterns from Hi-C contact maps.Multi-scale Conditioning: Incorporates biological priors ($y_{TAD}$, $y_{AB}$, $y_{Loop}$) into the latent space to ensure the generated structures adhere to known genomic organization rules.Bio-Physical Constraint Loss:Reconstruction Loss: Fits the Hi-C frequency-to-distance relationship.Excluded Volume (EV) Loss: Penalizes non-physical overlapping of DNA beads.Chain Continuity Loss: Ensures stable Euclidean distance between adjacent genomic bins.Ensemble Inference: Generates structural ensembles to represent chromatin dynamics and utilizes the Kabsch Algorithm for structural alignment and consensus generation.

Quick Start
1. Environment Setup
Bash
git clone https://github.com/YourUsername/GM12878-3D-CVAE.git
cd GM12878-3D-CVAE
pip install -r requirements.txt
2. Data Preparation
Place your GM12878 .mcool or .hic files in the data/ directory. Ensure you have the corresponding bedpe files for TAD/Loop labels if training from scratch.

3. Model Training
Bash
python main.py --epochs 100 --batch_size 64 --lr 2e-4
4. Structural Inference
Generate a 3D ensemble and a consensus PDB file:

Bash
python inference.py --checkpoint ./outputs/best_model.pt --samples 50
📊 Visualization & Evaluation
The generated consensus_structure.pdb can be visualized using standard molecular modeling software such as PyMOL, VMD, or ChimeraX.

Recommended PyMOL commands:

Bash
load results/consensus_structure.pdb
as tube
spectrum count, rainbow