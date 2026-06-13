# Reference-Free Noise Calibration for Single-Qubit Channels

**Bridging continuous-variable quantum optics to gate-model quantum computing**

A self-consistent Pauli Transfer Matrix (PTM) reconstruction framework for characterizing unknown noisy quantum channels — inspired by the master's thesis *"Balanced Homodyne Detection without a Coherent State Local Oscillator"*.

---

## 📌 Overview

In quantum computing, every gate or idle period applies an **unknown noisy channel** to qubits. Traditional noise characterization often relies on perfectly trusted reference instruments or detailed noise models. This project demonstrates a **reference-free, self-consistent calibration method** that removes this assumption.

By adapting techniques from continuous-variable quantum optics (specifically the thesis methodology), this simulation shows how to:
- Reconstruct the noise channel’s effect **without assuming a specific noise model**.
- Use only relative measurements across a small set of calibration states.
- Detect when the noise cannot be described by a simple linear (Markovian) map.

This approach is directly applicable to **superconducting qubits**, **trapped-ion systems**, and other NISQ hardware.

## 🎯 Key Features

- **Self-Consistent PTM Reconstruction**: Estimates the full 4×4 Pauli Transfer Matrix using linear inversion over six calibration states (`±X`, `±Y`, `±Z`).
- **D-Criterion Diagnostic**: A powerful scalar metric that flags non-Markovian or state-dependent errors (e.g., crosstalk, calibration drift).
- **Shot-Noise Realism**: Simulates finite-shot measurements, showing how performance scales with measurement budget.
- **Comparative Analysis**: Benchmarks against ground-truth and naive "ideal instrument" assumptions.
- **Non-Markovian Stress Test**: Demonstrates how the D-criterion dramatically increases when simple linear models fail.

## 🔬 Methodology

1. **True Noisy Channel** — Simulated via QuTiP Lindblad master equation (amplitude damping + dephasing).
2. **Calibration States** — Six eigenstates of the Pauli operators (trusted preparation).
3. **Measurement** — Projective Pauli measurements with binomial shot noise.
4. **Reconstruction** — Linear least-squares to recover the Pauli Transfer Matrix.
5. **Validation** — D-criterion residual + comparison with ground truth.

## 📊 Results Highlights

- **Reconstruction Accuracy**: Self-consistent method achieves ~8× lower error than assuming no noise.
- **Measurement Efficiency**: Useful calibration achieved with only a few hundred shots per state.
- **Diagnostic Power**: D-criterion increases by **~200×** when state-dependent (non-Markovian) errors are present.
- Visualizations include PTM heatmaps, Bloch sphere trajectories, scaling studies, and diagnostic bar plots.

## 🚀 Real-World Advantages in Quantum Computing

This reference-free calibration technique offers several **practical benefits** for NISQ-era quantum computing:

### 1. **Reduced Calibration Overhead**
- Eliminates the need for perfectly characterized reference states or instruments.
- Enables **frequent, lightweight recalibration** during long computations.

### 2. **Improved Error Mitigation**
- Provides a calibrated Pauli Transfer Matrix that can be directly used in:
  - Zero-Noise Extrapolation (ZNE)
  - Probabilistic Error Cancellation
  - Customized error models for VQE, QAOA, or other algorithms.

### 3. **Better Detection of Complex Noise**
- The D-criterion acts as an early-warning system for:
  - Crosstalk
  - State-preparation errors
  - Non-Markovian effects (e.g., from environmental memory or drifting control parameters)
- Helps decide when a simple noise model is no longer sufficient.

### 4. **Hardware Agnostic & Scalable**
- Works with superconducting, trapped-ion, or neutral-atom platforms.
- Foundation for multi-qubit extensions (crosstalk mapping).
- Can be integrated with Qiskit, Cirq, or other frameworks for real-hardware deployment.

### 5. **Efficiency Gains**
By using self-consistent reconstruction instead of assuming ideal behavior, circuits can be optimized with more accurate noise-aware compilation, leading to **higher fidelity** and **longer effective circuit depth** on noisy hardware.

## 🛠️ Installation & Usage

### Requirements
```bash
pip install qutip numpy matplotlib
