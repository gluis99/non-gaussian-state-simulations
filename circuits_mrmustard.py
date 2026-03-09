import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

from mrmustard import settings
from mrmustard.lab import SqueezedVacuum, Number, BSgate
from mrmustard.physics.wigner import wigner_discretized

r = lambda r_dB: r_dB / 20 * np.log(10)

# Helper function to display Wigner functions in a GitHub-friendly way
def show_state_wigner(
    state,
    title="State Wigner Function",
    q_min=-5,
    q_max=5,
    n_points=301,
    fock_cutoff=60,
):
    """Render a static Wigner plot that is visible on GitHub."""
    q = np.linspace(q_min, q_max, n_points)
    p = np.linspace(q_min, q_max, n_points)
    rho = state.dm().fock_array(fock_cutoff)
    W, X, P = wigner_discretized(rho, q, p)
    absmax = np.max(np.abs(W))
    if absmax == 0:
        absmax = 1e-12
    norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.pcolormesh(X, P, W, shading="auto", cmap="RdBu_r", norm=norm)
    fig.colorbar(c, ax=ax, label="W(x,p)")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# Circuits for generating grid-like states using the MRMustard library.


#################################################################################################
# Cat state generation circuit
# Following J. Hastrup et al. (2020). https://doi.org/10.1364/OL.383194

# Two options are supported:
 # 1) Provide only r0 -> r1=-r0 is used
 # 2) Provide all two squeezing parameters explicitly
def circuit_2cat(r0, r1 = None, n = 1, cutoff = None, r_in_dB = False):

    if r_in_dB:
        r0 = r(r0)
        if r1 is not None:
            r1 = r(r1)

    if r1 == r0:
        raise ValueError("r1 cannot be equal to r0")
    if r1 is None:
        r1 = -r0
    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")
    
    theta = np.arcsin(np.sqrt((1-np.exp(2*r1))/((np.exp(2*r0)-np.exp(2*r1)))))

    if cutoff is None:
        out = (
            SqueezedVacuum(0, r0, phi=0)
            >> SqueezedVacuum(1, r1, phi=0)
            >> BSgate((0, 1), theta=theta, phi=0)
            >> Number(0, n).dual
        )
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = (
                SqueezedVacuum(0, r0, phi=0)
                >> SqueezedVacuum(1, r1, phi=0)
                >> BSgate((0, 1), theta=theta, phi=0)
                >> Number(0, n).dual
            )
    probability = out.probability

    return out.normalize(), probability


def circuit_4cat(Ns, r0, r2, r1=None, r3=None, n2=0, cutoff=None, r_in_dB=False):
    if not isinstance(Ns, (list, tuple)) or len(Ns) != 2:
        raise ValueError("Ns must be a list/tuple with exactly two elements: [n1, n3]")
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in Ns):
        raise ValueError("Ns entries must be non-negative integers")
    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")
    
    if r_in_dB:
        r0 = r(r0)
        r2 = r(r2)
        if r1 is not None:
            r1 = r(r1)
        if r3 is not None:
            r3 = r(r3)

    explicit_r1 = r1 is not None
    explicit_r3 = r3 is not None

    if not explicit_r1:
        r1 = -r0
    if not explicit_r3:
        r3 = -r2

    def theta(ra, rb, label):
        denom = np.exp(2 * ra) - np.exp(2 * rb)
        if np.isclose(denom, 0.0):
            raise ValueError(f"Invalid parameters for {label}: denominator is zero")

        ratio = (1 - np.exp(2 * rb)) / denom
        if ratio < -1e-12 or ratio > 1 + 1e-12:
            raise ValueError(
                f"Invalid parameters for {label}: arcsin argument out of range (got {ratio})"
            )
        ratio = np.clip(ratio, 0.0, 1.0)
        return np.arcsin(np.sqrt(ratio))

    theta1 = theta(r0, r1, "BS1")
    theta2 = theta(r2, r3, "BS2")

    if cutoff is None:
        out = (
            SqueezedVacuum(0, r0, phi=0)
            >> SqueezedVacuum(1, r1, phi=0)
            >> SqueezedVacuum(2, r2, phi=0)
            >> SqueezedVacuum(3, r3, phi=0)
            >> BSgate((0, 1), theta=theta1, phi=0)
            >> BSgate((2, 3), theta=theta2, phi=0)
            >> BSgate((0, 2), theta=np.pi / 4, phi=np.pi / 2)
            >> Number(1, Ns[0]).dual
            >> Number(2, n2).dual
            >> Number(3, Ns[1]).dual
        )
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = (
                SqueezedVacuum(0, r0, phi=0)
                >> SqueezedVacuum(1, r1, phi=0)
                >> SqueezedVacuum(2, r2, phi=0)
                >> SqueezedVacuum(3, r3, phi=0)
                >> BSgate((0, 1), theta=theta1, phi=0)
                >> BSgate((2, 3), theta=theta2, phi=0)
                >> BSgate((0, 2), theta=np.pi / 4, phi=np.pi / 2)
                >> Number(1, Ns[0]).dual
                >> Number(2, n2).dual
                >> Number(3, Ns[1]).dual
            )
    out_norm = out.normalize()

    return out_norm, out.probability


def circuit_4cat_v2(Ns, r0, r2, r1=None, r3=None, n2=0, cutoff=None, r_in_dB=False):

    if not isinstance(Ns, (list, tuple)) or len(Ns) != 2:
        raise ValueError("Ns must be a list/tuple with exactly two elements: [n1, n3]")
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in Ns):
        raise ValueError("Ns entries must be non-negative integers")
    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")

    if r_in_dB:
        r0 = r(r0)
        r2 = r(r2)
        if r1 is not None:
            r1 = r(r1)
        if r3 is not None:
            r3 = r(r3)

    if r1 is None:
        r1 = -r0
    if r3 is None:
        r3 = -r2

    def theta(ra, rb, label):
        denom = np.exp(2 * ra) - np.exp(2 * rb)
        if np.isclose(denom, 0.0):
            raise ValueError(f"Invalid parameters for {label}: denominator is zero")

        ratio = (1 - np.exp(2 * rb)) / denom
        if ratio < -1e-12 or ratio > 1 + 1e-12:
            raise ValueError(
                f"Invalid parameters for {label}: arcsin argument out of range (got {ratio})"
            )
        ratio = np.clip(ratio, 0.0, 1.0)
        return np.arcsin(np.sqrt(ratio))

    n1, n3 = Ns

    theta1 = theta(r0, r1, "BS1")
    theta2 = theta(r2, r3, "BS2")

    if cutoff is None:
        out = (
            SqueezedVacuum(0, r0, phi=0)
            >> SqueezedVacuum(1, r1, phi=0)
            >> SqueezedVacuum(2, r2, phi=0)
            >> SqueezedVacuum(3, r3, phi=0)
            >> BSgate((0, 1), theta=theta1, phi=0)
            >> Number(0, n1).dual
            >> BSgate((2, 3), theta=theta2, phi=0)
            >> Number(2, n3).dual
            >> BSgate((1, 3), theta=np.pi / 4, phi=np.pi / 2)
            >> Number(3, n2).dual
        )
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = (
                SqueezedVacuum(0, r0, phi=0)
                >> SqueezedVacuum(1, r1, phi=0)
                >> SqueezedVacuum(2, r2, phi=0)
                >> SqueezedVacuum(3, r3, phi=0)
                >> BSgate((0, 1), theta=theta1, phi=0)
                >> Number(0, n1).dual
                >> BSgate((2, 3), theta=theta2, phi=0)
                >> Number(2, n3).dual
                >> BSgate((1, 3), theta=np.pi / 4, phi=np.pi / 2)
                >> Number(3, n2).dual
            )
    out_norm = out.normalize()

    return out_norm, out.probability

##################################################################################################
# Gaussian boson sampling like circuit for generating grid-like states
# Like in M. V. Larsen et al. (2025). https://doi.org/10.1038/s41586-025-09044-5

# Like in TN_Sampling/Beework2.0_example.ipynb
def circuit_3mode_GBS_original(
    Ns, 
    r0, r1=None, r2=None,
    theta21=np.pi/4, phi21=0,
    theta10=np.pi/6, phi10=0,
    cutoff=None,
    r_in_dB=False,
):

    if not isinstance(Ns, (list, tuple)) or len(Ns) != 2:
        raise ValueError("Ns must be a list/tuple with exactly two elements: [n0, n1]")
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in Ns):
        raise ValueError("Ns entries must be non-negative integers")

    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")

    # Two squeezing options: provide only r0 (defaults r1=r2=r0) or provide all explicitly.
    if r1 is None:
        r1 = r0
    if r2 is None:
        r2 = r0

    if r_in_dB:
        r0 = r(r0)
        if r1 is not None:
            r1 = r(r1)
        if r2 is not None:
            r2 = r(r2)

    if cutoff is None:
        out = (
            SqueezedVacuum(0, r0, phi=0)
            >> SqueezedVacuum(1, r1, phi=0)
            >> SqueezedVacuum(2, -r2, phi=0)
            >> BSgate((2, 1), theta=theta21, phi=phi21)
            >> BSgate((1, 0), theta=theta10, phi=phi10)
            >> Number(0, Ns[0]).dual
            >> Number(1, Ns[1]).dual
        )
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = (
                SqueezedVacuum(0, r0, phi=0)
                >> SqueezedVacuum(1, r1, phi=0)
                >> SqueezedVacuum(2, -r2, phi=0)
                >> BSgate((2, 1), theta=theta21, phi=phi21)
                >> BSgate((1, 0), theta=theta10, phi=phi10)
                >> Number(0, Ns[0]).dual
                >> Number(1, Ns[1]).dual
            )
    probability = out.probability
    #print(f'Probability: {out.probability}')

    return out.normalize(), probability

# Generates output state for the 3 input modes circuit
def circuit_3mode_GBS(
    Ns, 
    r0, r1=None, r2=None,
    theta0=np.pi/4, phi0=0,
    theta1=np.pi/4, phi1=0,
    cutoff=None,
    r_in_dB=False,
):

    if not isinstance(Ns, (list, tuple)) or len(Ns) != 2:
        raise ValueError("Ns must be a list/tuple with exactly two elements: [n0, n1]")
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in Ns):
        raise ValueError("Ns entries must be non-negative integers")

    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")

    # Two squeezing options: provide only r0 (defaults r1=r2=r0) or provide all explicitly.
    if r1 is None:
        r1 = r0
    if r2 is None:
        r2 = r0

    if r_in_dB:
        r0 = r(r0)
        if r1 is not None:
            r1 = r(r1)
        if r2 is not None:
            r2 = r(r2)

    if cutoff is None:
        out = (
            SqueezedVacuum(0, r0, phi=np.pi)
            >> SqueezedVacuum(1, r1, phi=np.pi)
            >> SqueezedVacuum(2, r2, phi=0)
            >> BSgate((0, 1), theta=theta0, phi=phi0)
            >> BSgate((1, 2), theta=theta1, phi=phi1)
            >> Number(0, Ns[0]).dual
            >> Number(1, Ns[1]).dual
        )
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = (
                SqueezedVacuum(0, r0, phi=np.pi)
                >> SqueezedVacuum(1, r1, phi=np.pi)
                >> SqueezedVacuum(2, r2, phi=0)
                >> BSgate((0, 1), theta=theta0, phi=phi0)
                >> BSgate((1, 2), theta=theta1, phi=phi1)
                >> Number(0, Ns[0]).dual
                >> Number(1, Ns[1]).dual
            )
    probability = out.probability

    return out.normalize(), probability

# Generalization to N modes (breaks down )
def circuit_Nmode_GBS(
    N,
    Ns,
    squeezing_params=None,
    r_rest=0,
    bs_params=None,     # list of (j, theta, phi)
    cutoff=None,
    r_in_dB=False,
):

    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2")

    if not isinstance(Ns, (list, tuple)):
        raise ValueError("Ns must be a list/tuple of (j, n_j) pairs")

    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")

    def _validate_indexed_entries(entries, entry_len, label):
        if entries is None:
            return []
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"{label} must be a list/tuple")
        validated = []
        for item in entries:
            if not isinstance(item, (list, tuple)) or len(item) != entry_len:
                raise ValueError(
                    f"Each {label} entry must be a tuple of length {entry_len}"
                )
            j = item[0]
            if not isinstance(j, (int, np.integer)):
                raise ValueError(f"{label} indices j must be integers")
            validated.append(tuple(item))
        return validated

    Ns = _validate_indexed_entries(Ns, 2, "Ns")
    bs_params = _validate_indexed_entries(bs_params, 3, "bs_params")

    # squeezing_params must be a list of tuples (j, r_j).
    # Unspecified modes use r_rest.
    if squeezing_params is None:
        squeezing_entries = []
    else:
        squeezing_entries = _validate_indexed_entries(squeezing_params, 2, "squeezing_params")

    squeezing_by_mode = {j: r_rest for j in range(N)}
    seen_squeezing = set()
    for j, r_j in squeezing_entries:
        if not (0 <= j <= N - 1):
            raise ValueError(f"squeezing_params index j out of range: {j}. Expected 0..{N - 1}")
        if j in seen_squeezing:
            raise ValueError(f"Duplicate squeezing_params entry for mode j={j}")
        seen_squeezing.add(j)
        squeezing_by_mode[j] = r_j

    squeezing_params = [squeezing_by_mode[j] for j in range(N)]

    max_valid_j = N - 2

    n_by_mode = {j: 0 for j in range(N - 1)}
    seen_ns = set()
    for j, n_j in Ns:
        if not (0 <= j <= max_valid_j):
            raise ValueError(f"Ns index j out of range: {j}. Expected 0..{max_valid_j}")
        if not isinstance(n_j, (int, np.integer)) or n_j < 0:
            raise ValueError("Ns values n_j must be non-negative integers")
        if j in seen_ns:
            raise ValueError(f"Duplicate Ns entry for mode j={j}")
        seen_ns.add(j)
        n_by_mode[j] = int(n_j)

    bs_by_index = {j: (np.pi / 4, 0.0) for j in range(N - 1)}
    seen_bs = set()
    for j, theta_j, phi_j in bs_params:
        if not (0 <= j <= max_valid_j):
            raise ValueError(f"bs_params index j out of range: {j}. Expected 0..{max_valid_j}")
        if j in seen_bs:
            raise ValueError(f"Duplicate bs_params entry for j={j}")
        seen_bs.add(j)
        bs_by_index[j] = (theta_j, phi_j)

    if r_in_dB:
        squeezing_params = [r(sq) for sq in squeezing_params]

    
    input_state = [
        SqueezedVacuum(i, sq, phi=(0 if i % 2 == 1 else np.pi/2))
        for i, sq in enumerate(squeezing_params)
    ]

    # Single chain of adjacent beam splitters: (0,1), (1,2), ..., (N-2,N-1).
    bs_gates = []
    for i in range(N - 1):
        theta, phi = bs_by_index[i]
        bs_gates.append(BSgate((i, i + 1), theta=theta, phi=phi))

    interferometer = bs_gates[0]
    for gate in bs_gates[1:]:
        interferometer >>= gate

    measurement = [
        Number(i, n_by_mode[i]).dual for i in range(N - 1)
    ]

    if cutoff is None:
        out = input_state[0]
        for state in input_state[1:]:
            out = out >> state
        out = out >> interferometer
        for proj in measurement:
            out = out >> proj
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = input_state[0]
            for state in input_state[1:]:
                out = out >> state
            out = out >> interferometer
            for proj in measurement:
                out = out >> proj
    probability = out.probability

    return out.normalize(), probability

##################################################################################################
# Side-by-side Wigner comparison plotting function
def compare_Wigners(
    Wigner_1,
    Wigner_2,
    title1="State 1",
    title2="State 2",
    cmap="RdBu_r",
    percentile=99.5,
    figsize=(11, 4.8),
    common_scale=True,
    shared_colorbar=True,
    cbar_label=r"$W(q,p)$",
    savepath=None,
    dpi=400,
):
    W1, X1, P1 = Wigner_1
    W2, X2, P2 = Wigner_2

    assert W1.shape == (len(P1), len(X1)), "Wigner_1 shape mismatch."
    assert W2.shape == (len(P2), len(X2)), "Wigner_2 shape mismatch."

    extent1 = (X1.min(), X1.max(), P1.min(), P1.max())
    extent2 = (X2.min(), X2.max(), P2.min(), P2.max())

    v1 = np.percentile(np.abs(W1), percentile)
    v2 = np.percentile(np.abs(W2), percentile)
    if not np.isfinite(v1) or v1 <= 0:
        v1 = np.max(np.abs(W1))
    if not np.isfinite(v2) or v2 <= 0:
        v2 = np.max(np.abs(W2))

    if common_scale:
        vmax = max(v1, v2)
        norm1 = norm2 = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm1 = mcolors.TwoSlopeNorm(vmin=-v1, vcenter=0.0, vmax=v1)
        norm2 = mcolors.TwoSlopeNorm(vmin=-v2, vcenter=0.0, vmax=v2)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    im0 = axes[0].imshow(
        W1, extent=extent1, origin="lower", cmap=cmap, norm=norm1, interpolation="none"
    )
    im1 = axes[1].imshow(
        W2, extent=extent2, origin="lower", cmap=cmap, norm=norm2, interpolation="none"
    )

    for ax, title in zip(axes, [title1, title2]):
        ax.set_title(title)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$p$")
        ax.set_aspect("equal")
        ax.tick_params(direction="in", top=True, right=True)

    if shared_colorbar:
        mappable = im0 if common_scale else im1
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
    else:
        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar0.set_label(cbar_label)
        cbar1.set_label(cbar_label)

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_wigner_orthogonal_cuts(
    Wigner,
    x_cut=0.0,
    p_cut=0.0,
    prominence_frac=0.02,
    min_separation_pts=8,
    figsize=(10, 5),
    title=r"Orthogonal Wigner cuts: $W(X,p_0)$ vs $W(x_0,P)$",
    curve_colors=("tab:blue", "tab:orange"),
    extrema_colors=("tab:blue", "tab:orange"),
    extrema_linestyle=":",
    extrema_alpha=0.45,
    show_grid=True,
    savepath=None,
    dpi=300,
):
    """Plot W(X, p_cut) and W(x_cut, P) with extrema shown as vertical lines.

    Args:
        Wigner: Tuple (W, X, P) from ``wigner_discretized``.
        x_cut: X value used for the vertical cut W(x_cut, P).
        p_cut: P value used for the horizontal cut W(X, p_cut).
        prominence_frac: Peak prominence as a fraction of slice amplitude.
        min_separation_pts: Minimum sample separation between neighboring extrema.

    Returns:
        dict with axes, slices, extrema positions, and alignment diagnostics.
    """

    W, X, P = Wigner
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array")

    def _extract_axis(arr, target_len, label):
        arr = np.asarray(arr)

        if arr.ndim == 1:
            if arr.size == target_len and np.ptp(arr) > 0:
                return arr
            unique = np.unique(np.round(arr, 12))
            if unique.size == target_len and np.ptp(unique) > 0:
                return unique

        if arr.ndim == 2:
            candidates = [arr[0, :], arr[:, 0], arr[-1, :], arr[:, -1]]
            for c in candidates:
                c = np.asarray(c)
                if c.size == target_len and np.ptp(c) > 0:
                    return c

        raise ValueError(
            f"Could not infer 1D {label} axis from shape {arr.shape}. "
            "Expected vector, meshgrid, or flattened meshgrid values."
        )

    def _nearest_distances(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.array([])
        return np.min(np.abs(a[:, None] - b[None, :]), axis=1)

    x_axis = _extract_axis(X, W.shape[1], "X")
    p_axis = _extract_axis(P, W.shape[0], "P")

    # Keep axis orientation increasing and rotate W consistently.
    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis[::-1]
        W = W[:, ::-1]
    if p_axis[0] > p_axis[-1]:
        p_axis = p_axis[::-1]
        W = W[::-1, :]

    ix_cut = int(np.argmin(np.abs(x_axis - x_cut)))
    ip_cut = int(np.argmin(np.abs(p_axis - p_cut)))

    cut_x = W[ip_cut, :]   # W(X, p_cut)
    cut_p = W[:, ix_cut]   # W(x_cut, P)

    def _slice_extrema(coords, values):
        amp = float(np.max(values) - np.min(values))
        prominence = max(prominence_frac * amp, 1e-12)
        max_idx, _ = find_peaks(values, prominence=prominence, distance=min_separation_pts)
        min_idx, _ = find_peaks(-values, prominence=prominence, distance=min_separation_pts)
        return {
            "max_idx": max_idx,
            "min_idx": min_idx,
            "max_pos": coords[max_idx],
            "min_pos": coords[min_idx],
        }

    ext_x = _slice_extrema(x_axis, cut_x)
    ext_p = _slice_extrema(p_axis, cut_p)

    dmax = _nearest_distances(ext_x["max_pos"], ext_p["max_pos"])
    dmin = _nearest_distances(ext_x["min_pos"], ext_p["min_pos"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        x_axis,
        cut_x,
        lw=2,
        color=curve_colors[0],
        label=rf"$W(X, P={p_axis[ip_cut]:.3g})$",
    )
    ax.plot(
        p_axis,
        cut_p,
        lw=2,
        color=curve_colors[1],
        label=rf"$W(X={x_axis[ix_cut]:.3g}, P)$",
    )

    extrema_x = np.sort(np.concatenate([ext_x["max_pos"], ext_x["min_pos"]]))
    extrema_p = np.sort(np.concatenate([ext_p["max_pos"], ext_p["min_pos"]]))

    for xpos in extrema_x:
        ax.axvline(
            xpos,
            color=extrema_colors[0],
            ls=extrema_linestyle,
            lw=1.1,
            alpha=extrema_alpha,
        )
    for xpos in extrema_p:
        ax.axvline(
            xpos,
            color=extrema_colors[1],
            ls=extrema_linestyle,
            lw=1.1,
            alpha=extrema_alpha,
        )

    # Proxy handles for clean legend entries for dotted extrema lines.
    ax.plot([], [], ls=extrema_linestyle, color=extrema_colors[0],
            label=r"extrema of $W(X,p_0)$")
    ax.plot([], [], ls=extrema_linestyle, color=extrema_colors[1],
            label=r"extrema of $W(x_0,P)$")

    ax.axvline(0.0, color="gray", ls="--", lw=1)
    if show_grid:
        ax.grid(alpha=0.25)
    ax.set_xlabel("Quadrature value")
    ax.set_ylabel("Wigner value")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()

    diagnostics = {
        "x_step": float(x_axis[1] - x_axis[0]),
        "p_step": float(p_axis[1] - p_axis[0]),
        "x_cut_actual": float(x_axis[ix_cut]),
        "p_cut_actual": float(p_axis[ip_cut]),
        "n_max_x_cut": int(len(ext_x["max_pos"])),
        "n_min_x_cut": int(len(ext_x["min_pos"])),
        "n_max_p_cut": int(len(ext_p["max_pos"])),
        "n_min_p_cut": int(len(ext_p["min_pos"])),
        "maxima_alignment_mean": float(np.mean(dmax)) if len(dmax) else None,
        "maxima_alignment_max": float(np.max(dmax)) if len(dmax) else None,
        "minima_alignment_mean": float(np.mean(dmin)) if len(dmin) else None,
        "minima_alignment_max": float(np.max(dmin)) if len(dmin) else None,
    }

    return {
        "x_axis": x_axis,
        "p_axis": p_axis,
        "cut_x": cut_x,
        "cut_p": cut_p,
        "extrema_x_cut": ext_x,
        "extrema_p_cut": ext_p,
        "diagnostics": diagnostics,
    }


