# Reference-Free Noise Calibration for Quantum Channels

**Bridging Continuous-Variable Quantum Optics to Gate-Model NISQ Hardware**

A series of self-consistent noise characterization tools inspired by the master's thesis  
*"Balanced Homodyne Detection without a Coherent State Local Oscillator"*.

---

## 📌 Project Overview

Accurate noise characterization is critical for reliable quantum computing, yet traditional methods often depend on perfectly trusted reference instruments. This repository demonstrates **reference-free, self-consistent calibration techniques** that eliminate this assumption.

The project has two main components:

### 1. Single-Qubit Calibration (`qubit_noise_calibration.py`)
Self-consistent Pauli Transfer Matrix (PTM) reconstruction for a single qubit using six calibration states. Includes a D-criterion diagnostic to detect non-Markovian or state-dependent errors.

### 2. Two-Qubit Superconducting Extension (`two-qubit/qubit_noise_calibration_2q.py`)
Extends the framework to two coupled qubits with realistic superconducting parameters (T₁, T₂, readout error) and introduces a **locality residual** for detecting static ZZ crosstalk.

Both implementations follow the same core philosophy from the thesis: reconstruct the noise channel from raw calibration statistics **without assuming a specific noise model**, then use scalar consistency diagnostics to validate simplifying assumptions.

---

## 🎯 Key Features

- **Self-Consistent PTM Reconstruction** — No trusted reference or assumed noise model
- **D-Criterion Diagnostic** — Flags when a single linear map fails to explain the data
- **Realistic Noise Models** — Lindblad dynamics + finite shot noise + readout errors
- **Crosstalk Detection** (2Q) — Locality gap identifies non-local dynamics
- **Measurement Scaling Studies** — Shows practical resource requirements

---

## 📊 Visualizations

### Single-Qubit Results
- **Figure 1**: Pauli Transfer Matrix Comparison (Ground Truth vs Reconstructed vs Naive)
- **Figure 2**: Bloch Sphere — Ideal vs Noisy Calibration States
- **Figure 3**: Reconstruction Error & D-Criterion vs Measurement Shots
- **Figure 4**: D-Criterion as Non-Markovianity Diagnostic (~200× increase)

### Two-Qubit Superconducting Results
- **Figure 5**: 16×16 PTM Comparison (Full vs Reconstructed vs Local)
- **Figure 6**: Crosstalk Fingerprint (PTM Residual: Full − Local)
- **Figure 7**: Locality Gap vs ZZ Coupling Strength (Crosstalk Sensor)

See [`figures/FIGURES.md`](figures/FIGURES.md) for detailed descriptions of all figures.

---

## 🚀 Real-World Impact

- Enables **lightweight, frequent recalibration** on NISQ hardware without perfect references
- Provides calibrated PTMs for improved error mitigation (ZNE, PEC, etc.)
- Detects crosstalk and non-Markovian effects that standard methods might miss
- Directly applicable to superconducting (fixed-frequency transmons) and trapped-ion platforms
- Serves as a strong foundation for multi-qubit crosstalk mapping and hardware-aware algorithm optimization

---

## 🛠️ Installation & Running

### Prerequisites
```bash
pip install qutip numpy matplotlib
