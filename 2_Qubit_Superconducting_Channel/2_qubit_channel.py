"""
Reference-Free Noise Calibration for Two-Qubit Superconducting Channels
==========================================================================

[Full original docstring preserved]
"""

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

np.random.seed(7)

# -----------------------------------------------------------------------
# [All functions remain identical to the original up to the plotting section]
# -----------------------------------------------------------------------

SC_PARAMS = {
    "T1": 80e-6,
    "T2": 50e-6,
    "gate_time": 300e-9,
    "zz_coupling_hz": 50e3,
    "readout_error": 0.02,
}


def t1_t2_to_rates(T1, T2):
    gamma_1 = 1.0 / T1
    gamma_phi = max(1.0 / T2 - 1.0 / (2 * T1), 0.0)
    return gamma_1, gamma_phi


PAULI_1Q = {
    "I": qt.qeye(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz(),
}
PAULI_LABELS_1Q = ["I", "X", "Y", "Z"]
PAULI_LABELS_2Q = ["".join(p) for p in itertools.product(PAULI_LABELS_1Q, repeat=2)]


def pauli_op_2q(label):
    return qt.tensor(PAULI_1Q[label[0]], PAULI_1Q[label[1]])


def rho_to_pauli_vector_2q(rho):
    return np.array([qt.expect(pauli_op_2q(lab), rho) for lab in PAULI_LABELS_2Q])


def single_qubit_cal_states():
    states = {
        "+Z": qt.basis(2, 0), "-Z": qt.basis(2, 1),
        "+X": (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
        "-X": (qt.basis(2, 0) - qt.basis(2, 1)).unit(),
        "+Y": (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),
        "-Y": (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit(),
    }
    return {k: qt.ket2dm(v) for k, v in states.items()}


def calibration_states_2q():
    sq = single_qubit_cal_states()
    chosen_labels = ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]
    cal = {}
    for l1 in chosen_labels:
        for l2 in chosen_labels:
            label = f"{l1}|{l2}"
            cal[label] = qt.tensor(sq[l1], sq[l2])
    return cal


def build_true_channel_2q(T1=SC_PARAMS["T1"], T2=SC_PARAMS["T2"],
                          gate_time=SC_PARAMS["gate_time"],
                          zz_coupling_hz=SC_PARAMS["zz_coupling_hz"],
                          include_zz=True):
    gamma_1, gamma_phi = t1_t2_to_rates(T1, T2)
    sm1 = qt.tensor(qt.sigmam(), qt.qeye(2))
    sm2 = qt.tensor(qt.qeye(2), qt.sigmam())
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

    c_ops = [
        np.sqrt(gamma_1) * sm1, np.sqrt(gamma_1) * sm2,
        np.sqrt(gamma_phi / 2.0) * sz1, np.sqrt(gamma_phi / 2.0) * sz2,
    ]

    if include_zz:
        J = 2 * np.pi * zz_coupling_hz
        H = (J / 4.0) * qt.tensor(qt.sigmaz(), qt.sigmaz())
    else:
        H = 0 * qt.tensor(qt.qeye(2), qt.qeye(2))

    def apply_channel(rho):
        result = qt.mesolve(H, rho, [0, gate_time], c_ops=c_ops)
        return result.states[-1]
    return apply_channel


def measure_pauli_expectations_2q(rho, n_shots=4000, readout_error=SC_PARAMS["readout_error"]):
    results = {"II": 1.0}
    p_flip = readout_error
    for label in PAULI_LABELS_2Q:
        if label == "II":
            continue
        true_exp = qt.expect(pauli_op_2q(label), rho)
        n_nontrivial = sum(1 for c in label if c != "I")
        attenuation = (1 - 2 * p_flip) ** n_nontrivial
        attenuated_exp = true_exp * attenuation
        p_plus = np.clip((1 + attenuated_exp) / 2.0, 0, 1)
        counts_plus = np.random.binomial(n_shots, p_plus)
        est_exp = (2 * counts_plus - n_shots) / n_shots
        results[label] = est_exp
    return results


def reconstruct_ptm_2q(cal_states, measured_outputs):
    A, B = [], []
    for label, rho_in in cal_states.items():
        A.append(rho_to_pauli_vector_2q(rho_in))
        out = measured_outputs[label]
        B.append([out[lab] for lab in PAULI_LABELS_2Q])
    A = np.array(A)
    B = np.array(B)
    R_T, *_ = np.linalg.lstsq(A, B, rcond=None)
    R = R_T.T
    R[0, :] = 0
    R[0, 0] = 1
    return R, A, B


def d_criterion(R, A, B):
    residuals = B - A @ R.T
    D = np.mean(np.sum(residuals**2, axis=1))
    return D, residuals


def ground_truth_ptm_2q(channel):
    cal_states = calibration_states_2q()
    A, B = [], []
    for label, rho_in in cal_states.items():
        rho_out = channel(rho_in)
        A.append(rho_to_pauli_vector_2q(rho_in))
        B.append(rho_to_pauli_vector_2q(rho_out))
    A = np.array(A); B = np.array(B)
    R_T, *_ = np.linalg.lstsq(A, B, rcond=None)
    R = R_T.T
    R[0, :] = 0
    R[0, 0] = 1
    return R


def local_ptm_from_full(R_full):
    # [Full original implementation - unchanged]
    def idx(i, j): return 4 * i + j
    T = np.zeros((4, 4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    T[i, j, k, l] = R_full[idx(i, j), idx(k, l)]

    R_A = np.zeros((4, 4))
    R_B = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            R_A[i, k] = R_full[idx(i, 0), idx(k, 0)]
    for j in range(4):
        for l in range(4):
            R_B[j, l] = R_full[idx(0, j), idx(0, l)]
    if np.allclose(R_A, 0): R_A = np.eye(4)
    if np.allclose(R_B, 0): R_B = np.eye(4)

    for _ in range(50):
        num = np.einsum('ijkl,jl->ik', T, R_B)
        den = np.sum(R_B**2)
        R_A_new = num / den if den > 1e-12 else R_A
        num = np.einsum('ijkl,ik->jl', T, R_A_new)
        den = np.sum(R_A_new**2)
        R_B_new = num / den if den > 1e-12 else R_B
        if np.allclose(R_A_new, R_A, atol=1e-10) and np.allclose(R_B_new, R_B, atol=1e-10):
            break
        R_A, R_B = R_A_new, R_B_new

    if abs(R_A[0, 0]) > 1e-12:
        scale = R_A[0, 0]
        R_A = R_A / scale
        R_B = R_B * scale
    if abs(R_B[0, 0]) > 1e-12:
        scale = R_B[0, 0]
        R_B = R_B / scale
        R_A = R_A * scale

    return np.kron(R_A, R_B), R_A, R_B


def locality_residual(R_full, A, B):
    R_local, _, _ = local_ptm_from_full(R_full)
    residuals = B - A @ R_local.T
    D_local = np.mean(np.sum(residuals**2, axis=1))
    return D_local, R_local, residuals


def run_experiment_2q(n_shots=4000, include_zz=True, **sc_overrides):
    params = dict(SC_PARAMS)
    params.update(sc_overrides)
    channel = build_true_channel_2q(
        T1=params["T1"], T2=params["T2"], gate_time=params["gate_time"],
        zz_coupling_hz=params["zz_coupling_hz"], include_zz=include_zz
    )
    cal_states = calibration_states_2q()
    measured_outputs = {}
    for label, rho_in in cal_states.items():
        rho_out = channel(rho_in)
        measured_outputs[label] = measure_pauli_expectations_2q(
            rho_out, n_shots=n_shots, readout_error=params["readout_error"]
        )
    R_reconstructed, A, B = reconstruct_ptm_2q(cal_states, measured_outputs)
    D_full, _ = d_criterion(R_reconstructed, A, B)
    D_local, R_local, _ = locality_residual(R_reconstructed, A, B)
    R_true = ground_truth_ptm_2q(channel)
    return {
        "R_reconstructed": R_reconstructed, "R_true": R_true, "R_local": R_local,
        "D_full": D_full, "D_local": D_local, "A": A, "B": B,
        "measured_outputs": measured_outputs, "params": params,
    }


def matrix_error(R1, R2):
    return np.linalg.norm(R1 - R2, ord="fro")


def zz_coupling_scan(zz_values_hz, n_shots=4000, n_repeats=5):
    # [Full original implementation - unchanged for brevity]
    D_full_vals, D_full_std = [], []
    D_local_vals, D_local_std = [], []
    locality_gap_vals, locality_gap_std = [], []
    for zz in zz_values_hz:
        d_fulls, d_locals, gaps = [], [], []
        for _ in range(n_repeats):
            res = run_experiment_2q(n_shots=n_shots, include_zz=(zz > 0), zz_coupling_hz=zz)
            d_fulls.append(res["D_full"])
            d_locals.append(res["D_local"])
            gaps.append(res["D_local"] - res["D_full"])
        D_full_vals.append(np.mean(d_fulls))
        D_full_std.append(np.std(d_fulls))
        D_local_vals.append(np.mean(d_locals))
        D_local_std.append(np.std(d_locals))
        locality_gap_vals.append(np.mean(gaps))
        locality_gap_std.append(np.std(gaps))
    return {
        "zz_values_hz": zz_values_hz,
        "D_full": D_full_vals, "D_full_std": D_full_std,
        "D_local": D_local_vals, "D_local_std": D_local_std,
        "locality_gap": locality_gap_vals, "locality_gap_std": locality_gap_std,
    }


# Plotting functions (unchanged except output paths)
def plot_ptm_heatmaps_2q(R_true, R_reconstructed, R_local, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Ground-truth 16x16 PTM\n(with ZZ coupling)",
              "Self-consistently\nreconstructed PTM",
              "Best LOCAL (tensor-product)\napproximation"]
    mats = [R_true, R_reconstructed, R_local]
    vmax = max(np.max(np.abs(m)) for m in mats)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(16)); ax.set_xticklabels(PAULI_LABELS_2Q, rotation=90, fontsize=6)
        ax.set_yticks(range(16)); ax.set_yticklabels(PAULI_LABELS_2Q, fontsize=6)
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="PTM element value")
    fig.suptitle("Two-Qubit Pauli Transfer Matrix: Full vs Local Approximation", y=1.02)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zz_difference(R_true, R_local, outpath):
    diff = R_true - R_local
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.max(np.abs(diff))
    im = ax.imshow(diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(16)); ax.set_xticklabels(PAULI_LABELS_2Q, rotation=90, fontsize=7)
    ax.set_yticks(range(16)); ax.set_yticklabels(PAULI_LABELS_2Q, fontsize=7)
    ax.set_title("PTM Residual: Full - Local (Crosstalk Fingerprint)")
    fig.colorbar(im, label="Residual PTM element value")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zz_scan(scan_results, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    zz = np.array(scan_results["zz_values_hz"]) / 1e3
    axes[0].errorbar(zz, scan_results["D_full"], yerr=scan_results["D_full_std"], marker="o", label="Full 16x16", capsize=3)
    axes[0].errorbar(zz, scan_results["D_local"], yerr=scan_results["D_local_std"], marker="s", label="Local", capsize=3)
    axes[0].set_xlabel("Static ZZ coupling (kHz)"); axes[0].set_ylabel("D-criterion residual")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].errorbar(zz, scan_results["locality_gap"], yerr=scan_results["locality_gap_std"], marker="^", color="darkorange", capsize=3)
    axes[1].set_xlabel("Static ZZ coupling (kHz)"); axes[1].set_ylabel("Locality gap")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Superconducting qubit parameters:")
    for k, v in SC_PARAMS.items():
        print(f"  {k}: {v}")
    g1, gphi = t1_t2_to_rates(SC_PARAMS["T1"], SC_PARAMS["T2"])
    print(f"  -> gamma_1 = {g1:.2e} 1/s, gamma_phi = {gphi:.2e} 1/s")

    print("\nRunning two-qubit experiment WITH ZZ coupling (crosstalk)...")
    result = run_experiment_2q(n_shots=4000, include_zz=True)
    print(f"D_full:  {result['D_full']:.3e}")
    print(f"D_local: {result['D_local']:.3e}")
    print(f"Locality gap: {result['D_local'] - result['D_full']:.3e}")

    plot_ptm_heatmaps_2q(result["R_true"], result["R_reconstructed"], result["R_local"], "fig5_ptm_2q_heatmaps.png")
    plot_zz_difference(result["R_true"], result["R_local"], "fig6_zz_residual.png")

    print("\nRunning WITHOUT ZZ coupling (local noise)...")
    result_no_zz = run_experiment_2q(n_shots=4000, include_zz=False)
    print(f"D_full (no ZZ): {result_no_zz['D_full']:.3e}")

    print("\nScanning ZZ coupling strength...")
    zz_values = [0, 10e3, 50e3, 100e3, 150e3, 250e3, 400e3]
    scan = zz_coupling_scan(zz_values, n_shots=4000, n_repeats=8)
    plot_zz_scan(scan, "fig7_zz_scan.png")

    print("\n✅ All figures saved to current directory!")
    print("   fig5_ptm_2q_heatmaps.png")
    print("   fig6_zz_residual.png")
    print("   fig7_zz_scan.png")
