import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mrmustard import settings
from mrmustard.lab import Circuit, SqueezedVacuum, Number
from mrmustard.lab.transformations import BSgate
from mrmustard.physics.wigner import wigner_discretized

r = lambda r_dB: r_dB / 20 * np.log(10)

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

    input_state = [
        SqueezedVacuum(0, r0, phi=0),
        SqueezedVacuum(1, r1, phi=0)
    ]

    BS2 = BSgate([0,1], theta, phi=0)

    c = Circuit(input_state) >> BS2 >> Number(0, n).dual

    if cutoff is None:
        out = c.contract()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = c.contract()
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

    input_state = [
        SqueezedVacuum(0, r0, phi=0),
        SqueezedVacuum(1, r1, phi=0),
        SqueezedVacuum(2, r2, phi=0),
        SqueezedVacuum(3, r3, phi=0),
    ]

    BS1 = BSgate([0, 1], theta1, phi=0)
    BS2 = BSgate([2, 3], theta2, phi=0)
    BS3 = BSgate([0, 2], np.pi / 4, phi=np.pi / 2)

    interferometer = BS1 >> BS2 >> BS3

    measurement = [
        Number(1, Ns[0]).dual,
        Number(2, n2).dual,
        Number(3, Ns[1]).dual,
    ]

    c = Circuit(input_state) >> interferometer >> Circuit(measurement)
    if cutoff is None:
        out = c.contract()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = c.contract()
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

    cat1_circuit = Circuit([
        SqueezedVacuum(0, r0, phi=0),
        SqueezedVacuum(1, r1, phi=0),
    ]) >> BSgate([0, 1], theta1, phi=0) >> Number(0, n1).dual
    if cutoff is None:
        cat_1 = cat1_circuit.contract().normalize()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            cat_1 = cat1_circuit.contract().normalize()

    cat2_circuit = Circuit([
        SqueezedVacuum(2, r2, phi=0),
        SqueezedVacuum(3, r3, phi=0),
    ]) >> BSgate([2, 3], theta2, phi=0) >> Number(2, n3).dual
    if cutoff is None:
        cat_2 = cat2_circuit.contract().normalize()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            cat_2 = cat2_circuit.contract().normalize()

    BS = BSgate([1, 3], np.pi / 4, phi=np.pi / 2)
    measurement = [Number(3, n2).dual]

    c = Circuit([cat_1, cat_2]) >> BS >> Circuit(measurement)
    if cutoff is None:
        out = c.contract()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = c.contract()
    out_norm = out.normalize()

    return out_norm, out.probability

##################################################################################################
# Gaussian boson sampling like circuit for generating grid-like states
# Like in M. V. Larsen et al. (2025). https://doi.org/10.1038/s41586-025-09044-5

# Generates output state for the 3 input modes cascade circuit
def circuit_3mode_GBS(
    Ns, 
    r0, r1=None, r2=None,
    theta1=None, phi1=None,
    theta2=None, phi2=None,
    cutoff=None,
    r_in_dB=False,
):

    if not isinstance(Ns, (list, tuple)) or len(Ns) != 2:
        raise ValueError("Ns must be a list/tuple with exactly two elements: [n0, n1]")
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in Ns):
        raise ValueError("Ns entries must be non-negative integers")

    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")

    if any(v is None for v in (theta1, phi1, theta2, phi2)):
        raise ValueError("theta1, phi1, theta2, and phi2 must all be provided")

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

    input_state = [
        SqueezedVacuum(0, r0, phi=np.pi),
        SqueezedVacuum(1, r1, phi=np.pi),
        SqueezedVacuum(2, r2, phi=0)
    ]

    BS1 = BSgate([1,2], theta1, phi1)
    BS2 = BSgate([0,1], theta2, phi2)

    interferometer = BS1 >> BS2

    measurement = [
        Number(0, Ns[0]).dual,
        Number(1, Ns[1]).dual
    ]
    
    c = Circuit(input_state) >> interferometer >> Circuit(measurement)

    if cutoff is None:
        out = c.contract()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = c.contract()
    probability = out.probability
    #print(f'Probability: {out.probability}')

    return out.normalize(), probability

# Generalization to N modes cascade circuit with N-1 beam splitters
def circuit_Nmode_GBS(
    Ns,
    squeezing_params=None,
    bs_params=None,     # list of (theta, phi)
    cutoff=None,
    r_in_dB=False,
    r0=None,
):

    if not isinstance(Ns, (list, tuple)):
        raise ValueError("Ns must be a list/tuple of non-negative integers")
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in Ns):
        raise ValueError("Ns entries must be non-negative integers")

    if bs_params is None:
        raise ValueError("bs_params must be provided")

    if cutoff is not None and (not isinstance(cutoff, (int, np.integer)) or cutoff <= 0):
        raise ValueError("cutoff must be a positive integer")

    N = len(Ns) + 1

    # Two squeezing options:
    # 1) Provide only r0 -> all modes use r0
    # 2) Provide all squeezing parameters explicitly in squeezing_params
    if squeezing_params is None:
        if r0 is None:
            raise ValueError("Provide either r0 or squeezing_params")
        squeezing_params = [r0] * N
    elif np.isscalar(squeezing_params):
        squeezing_params = [squeezing_params] * N
    else:
        if len(squeezing_params) != N:
            raise ValueError(
                f"squeezing_params must have length N={N} (got {len(squeezing_params)})"
            )

    if N < 2:
        raise ValueError("Need at least 2 modes")
    if not isinstance(bs_params, (list, tuple)) or len(bs_params) != N - 1:
        raise ValueError(f"bs_params must be a list/tuple of length N-1 (expected {N - 1})")
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bs_params):
        raise ValueError("Each bs_params entry must be a (theta, phi) pair")

    if r_in_dB:
        squeezing_params = [r(sq) for sq in squeezing_params]

    
    input_state = [
        SqueezedVacuum(i, r, phi=(0 if i % 2 == 1 else np.pi/2))
        for i, r in enumerate(squeezing_params)
    ]

    # Single chain of adjacent beam splitters: (0,1), (1,2), ..., (N-2,N-1).
    bs_gates = []
    for i in range(N - 1):
        theta, phi = bs_params[i]
        bs_gates.append(BSgate([i, i + 1], theta, phi))

    interferometer = bs_gates[0]
    for gate in bs_gates[1:]:
        interferometer >>= gate

    measurement = [
        Number(i, n).dual for i, n in enumerate(Ns)
    ]

    c = Circuit(input_state) >> interferometer >> Circuit(measurement)
    if cutoff is None:
        out = c.contract()
    else:
        with settings(DEFAULT_FOCK_SIZE=cutoff, AUTOSHAPE_MIN=cutoff, AUTOSHAPE_MAX=cutoff):
            out = c.contract()
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

