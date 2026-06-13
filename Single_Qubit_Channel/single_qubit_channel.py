"""
Reference-Free Noise Calibration for Single-Qubit Channels
=============================================================

[Original detailed docstring preserved — explains the thesis bridge]
"""

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# [All functions remain exactly the same as original up to plotting section]

def build_true_channel(gamma_amp=0.08, gamma_phase=0.05, t_gate=1.0):
    sm = qt.sigmam()
    sz = qt.sigmaz()
    c_ops = [np.sqrt(gamma_amp) * sm, np.sqrt(gamma_phase / 2.0) * sz]
    H = 0 * qt.qeye(2)

    def apply_channel(rho):
        result = qt.mesolve(H, rho, [0, t_gate], c_ops=c_ops)
        return result.states[-1]
    return apply_channel


def calibration_states():
    states = {
        "+Z": qt.basis(2, 0), "-Z": qt.basis(2, 1),
        "+X": (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
        "-X": (qt.basis(2, 0) - qt.basis(2, 1)).unit(),
        "+Y": (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),
        "-Y": (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit(),
    }
    return {k: qt.ket2dm(v) for k, v in states.items()}


def measure_pauli_expectations(rho, n_shots=2000):
    results = {}
    for label, op in [("X", qt.sigmax()), ("Y", qt.sigmay()), ("Z", qt.sigmaz())]:
        true_exp = qt.expect(op, rho)
        p_plus = np.clip((1 + true_exp) / 2.0, 0, 1)
        counts_plus = np.random.binomial(n_shots, p_plus)
        est_exp = (2 * counts_plus - n_shots) / n_shots
        results[label] = est_exp
    return results


def rho_to_pauli_vector(rho):
    return np.array([1.0, qt.expect(qt.sigmax(), rho),
                     qt.expect(qt.sigmay(), rho), qt.expect(qt.sigmaz(), rho)])


def reconstruct_ptm(cal_states, measured_outputs):
    A, B = [], []
    for label, rho_in in cal_states.items():
        A.append(rho_to_pauli_vector(rho_in))
        out = measured_outputs[label]
        B.append([1.0, out["X"], out["Y"], out["Z"]])
    A = np.array(A)
    B = np.array(B)
    R_T, *_ = np.linalg.lstsq(A, B, rcond=None)
    R = R_T.T
    R[0, :] = [1, 0, 0, 0]
    return R, A, B


def d_criterion(R, A, B):
    residuals = B - A @ R.T
    D = np.mean(np.sum(residuals**2, axis=1))
    return D, residuals


def ground_truth_ptm(channel):
    basis_states = calibration_states()
    A, B = [], []
    for label, rho_in in basis_states.items():
        rho_out = channel(rho_in)
        A.append(rho_to_pauli_vector(rho_in))
        B.append(rho_to_pauli_vector(rho_out))
    A = np.array(A); B = np.array(B)
    R_T, *_ = np.linalg.lstsq(A, B, rcond=None)
    R = R_T.T
    R[0, :] = [1, 0, 0, 0]
    return R


def run_experiment(n_shots=2000, gamma_amp=0.08, gamma_phase=0.05):
    channel = build_true_channel(gamma_amp=gamma_amp, gamma_phase=gamma_phase)
    cal_states = calibration_states()
    measured_outputs = {}
    for label, rho_in in cal_states.items():
        rho_out = channel(rho_in)
        measured_outputs[label] = measure_pauli_expectations(rho_out, n_shots=n_shots)
    R_reconstructed, A, B = reconstruct_ptm(cal_states, measured_outputs)
    D, residuals = d_criterion(R_reconstructed, A, B)
    R_true = ground_truth_ptm(channel)
    R_naive = np.eye(4)
    return {
        "R_reconstructed": R_reconstructed, "R_true": R_true, "R_naive": R_naive,
        "D": D, "residuals": residuals, "A": A, "B": B,
        "measured_outputs": measured_outputs,
        "channel_params": {"gamma_amp": gamma_amp, "gamma_phase": gamma_phase},
    }


def matrix_error(R1, R2):
    return np.linalg.norm(R1 - R2, ord="fro")


def shot_scaling_study(shot_list, n_repeats=10, gamma_amp=0.08, gamma_phase=0.05):
    # [unchanged - full implementation as in original]
    errors_recon = []; errors_naive = []; D_values = []
    errors_recon_std = []; D_values_std = []
    for n_shots in shot_list:
        recon_errs = []; naive_errs = []; Ds = []
        for _ in range(n_repeats):
            res = run_experiment(n_shots=n_shots, gamma_amp=gamma_amp, gamma_phase=gamma_phase)
            recon_errs.append(matrix_error(res["R_reconstructed"], res["R_true"]))
            naive_errs.append(matrix_error(res["R_naive"], res["R_true"]))
            Ds.append(res["D"])
        errors_recon.append(np.mean(recon_errs))
        errors_recon_std.append(np.std(recon_errs))
        errors_naive.append(np.mean(naive_errs))
        D_values.append(np.mean(Ds))
        D_values_std.append(np.std(Ds))
    return {"shots": shot_list, "errors_recon": errors_recon, "errors_recon_std": errors_recon_std,
            "errors_naive": errors_naive, "D_values": D_values, "D_values_std": D_values_std}


def run_non_markovian_experiment(n_shots=4000, gamma_amp=0.05, gamma_phase=0.03, extra_rotation_strength=0.3):
    # [unchanged]
    channel = build_true_channel(gamma_amp=gamma_amp, gamma_phase=gamma_phase)
    cal_states = calibration_states()
    measured_outputs = {}
    for label, rho_in in cal_states.items():
        rho_eff = rho_in
        if label in ("+Z", "-Z"):
            sign = 1 if label == "+Z" else -1
            theta = sign * extra_rotation_strength
            U = (-1j * theta / 2 * qt.sigmax()).expm()
            rho_eff = U * rho_in * U.dag()
        rho_out = channel(rho_eff)
        measured_outputs[label] = measure_pauli_expectations(rho_out, n_shots=n_shots)
    R_reconstructed, A, B = reconstruct_ptm(cal_states, measured_outputs)
    D, residuals = d_criterion(R_reconstructed, A, B)
    R_true = ground_truth_ptm(channel)
    return {"R_reconstructed": R_reconstructed, "R_true": R_true, "D": D,
            "residuals": residuals, "A": A, "B": B}


# Plotting functions (unchanged except save paths in main)
def plot_ptm_comparison(R_true, R_reconstructed, R_naive, outpath):
    # [full original implementation]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["I", "X", "Y", "Z"]
    titles = ["Ground-truth PTM\n(noise-free readout)",
              "Self-consistently\nreconstructed PTM",
              "Naive 'ideal instrument'\nassumption (R = I)"]
    mats = [R_true, R_reconstructed, R_naive]
    vmax = max(np.max(np.abs(m)) for m in mats)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="PTM element value")
    fig.suptitle("Pauli Transfer Matrix: Ground Truth vs Reconstructed vs Naive Assumption", y=1.05)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shot_scaling(scaling_results, outpath):
    # [full original implementation - omitted for brevity]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    shots = scaling_results["shots"]
    axes[0].errorbar(shots, scaling_results["errors_recon"], yerr=scaling_results["errors_recon_std"],
                     marker="o", label="Self-consistent reconstruction", capsize=3)
    axes[0].axhline(scaling_results["errors_naive"][0], color="gray", linestyle="--",
                    label="Naive 'ideal instrument' (R = I)")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Number of measurement shots per Pauli/state")
    axes[0].set_ylabel("Frobenius-norm error vs ground-truth PTM")
    axes[0].set_title("Reconstruction accuracy vs measurement budget")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].errorbar(shots, scaling_results["D_values"], yerr=scaling_results["D_values_std"],
                     marker="s", color="darkorange", capsize=3)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of measurement shots per Pauli/state")
    axes[1].set_ylabel("D-criterion residual")
    axes[1].set_title("Self-consistency residual (D) vs measurement budget")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_d_criterion_comparison(D_markovian, D_non_markovian, outpath):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["Markovian\n(Lindblad-only)", "Non-Markovian\n(state-dependent error)"],
                   [D_markovian, D_non_markovian], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("D-criterion residual")
    ax.set_title("D-criterion as a self-consistency diagnostic")
    for bar, val in zip(bars, [D_markovian, D_non_markovian]):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.05, f"{val:.2e}", ha="center", fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bloch_vectors(cal_states, measured_outputs, channel, outpath):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for label, rho_in in cal_states.items():
        v_in = rho_to_pauli_vector(rho_in)[1:]
        out = measured_outputs[label]
        v_out = np.array([out["X"], out["Y"], out["Z"]])
        ax.scatter(*v_in, color="blue", s=40)
        ax.scatter(*v_out, color="red", s=40)
        ax.plot([v_in[0], v_out[0]], [v_in[1], v_out[1]], [v_in[2], v_out[2]],
                color="gray", linestyle="--", alpha=0.6)
        ax.text(*v_in, label, fontsize=8)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.cos(u) * np.sin(v); y = np.sin(u) * np.sin(v); z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.3, linewidth=0.5)
    ax.scatter([], [], color="blue", label="Ideal calibration states (input)")
    ax.scatter([], [], color="red", label="Noisy measured states (output)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Calibration states: ideal input vs. measured noisy output")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Running main experiment (Markovian Lindblad noise)...")
    result = run_experiment(n_shots=4000)
    print("\nGround-truth PTM:\n", np.round(result["R_true"], 3))
    print("\nReconstructed PTM:\n", np.round(result["R_reconstructed"], 3))
    print("\nNaive 'ideal instrument' PTM:\n", np.round(result["R_naive"], 3))
    print(f"\nD-criterion (Markovian case): {result['D']:.3e}")
    print(f"Frobenius error (reconstructed vs true): {matrix_error(result['R_reconstructed'], result['R_true']):.4f}")
    print(f"Frobenius error (naive vs true):         {matrix_error(result['R_naive'], result['R_true']):.4f}")

    # === MODIFIED: Save to current directory ===
    plot_ptm_comparison(result["R_true"], result["R_reconstructed"], result["R_naive"], "fig1_ptm_comparison.png")
    cal_states = calibration_states()
    channel = build_true_channel()
    plot_bloch_vectors(cal_states, result["measured_outputs"], channel, "fig2_bloch_vectors.png")

    print("\nRunning shot-scaling study...")
    scaling = shot_scaling_study([50, 100, 250, 500, 1000, 2000, 4000, 8000], n_repeats=15)
    plot_shot_scaling(scaling, "fig3_shot_scaling.png")

    print("\nRunning non-Markovian stress test...")
    nm_result = run_non_markovian_experiment()
    print(f"D-criterion (non-Markovian case): {nm_result['D']:.3e}")
    print(f"D-criterion ratio (non-Markovian / Markovian): {nm_result['D'] / result['D']:.1f}x")

    plot_d_criterion_comparison(result["D"], nm_result["D"], "fig4_d_criterion.png")

    print("\n✅ All figures saved to the current working directory!")
    print("   fig1_ptm_comparison.png")
    print("   fig2_bloch_vectors.png")
    print("   fig3_shot_scaling.png")
    print("   fig4_d_criterion.png")
