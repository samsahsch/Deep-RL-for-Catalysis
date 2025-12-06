# Neural Network Quantum Molecular Dynamics for Excited State Water Systems

This project implements a **Neural Network Quantum Molecular Dynamics (NNQMD)** framework designed to simulate nanoscale chemical dynamics in excited states. By bridging the accuracy of Density Functional Theory (DFT) with the efficiency of classical Molecular Dynamics (MD), we investigate the structural reorganization of water under hole-doped conditions (6e, 10e, 16e) relevant to heterogeneous catalysis and ultrafast electron diffraction (UED) experiments.

## 🧪 Overview

Understanding light-matter interactions and catalysis requires modeling systems in excited electronic states. Traditional DFT scales poorly ($O(N^3)$), while classical MD cannot capture bond breaking in excited states. 

This framework utilizes **E(3)-Equivariant Graph Neural Networks** to approximate the Potential Energy Surface (PES) of hole-doped water systems. We train models on high-fidelity DFT data to predict energy ($E$) and forces ($F = -\nabla E$), enabling stable MD simulations of ultrafast chemical dynamics.

## ✨ Key Features
* **Multi-Model Support:** Implementations/Interfaces for **NequIP**, **MACE**, and **Allegro**.
* **Excited State Dynamics:** specialized training regimes for Ground State (GS) and Charged States (6, 10, 16 electron holes).
* **MD Integration:** Interface with the **Atomistic Simulation Environment (ASE)** for running Langevin dynamics.
* **Structural Analysis:** Tools to compute Radial Distribution Functions (RDF) and $\Delta$ PDF to visualize bond softening and lattice expansion.

## 🧠 Architectures Compared

1.  **NequIP:** Uses E(3)-equivariant interaction blocks with tensor products.
2.  **MACE:** Multi-Atomic Cluster Expansion using higher-order message passing.
3.  **Allegro:** Strictly local equivariant architecture for massive parallelization.

## 💻 Installation

To replicate the environment used on NERSC Perlmutter (A100 GPUs), ensure you have the following dependencies:

```bash
# Clone the repository
git clone [https://github.com/samsahsch/Deep-RL-for-Catalysis.git](https://github.com/samsahsch/Deep-RL-for-Catalysis.git)
cd Deep-RL-for-Catalysis

# Recommended: Create a conda environment
conda create -n nnqmd python=3.9
conda activate nnqmd

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Model Libraries & Utilities
pip install nequip
pip install mace-torch
pip install allegro
pip install ase
pip install matplotlib numpy scipy