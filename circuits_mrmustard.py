import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mrmustard import settings
from mrmustard.lab import Circuit, SqueezedVacuum, Number
from mrmustard.lab.transformations import BSgate
from mrmustard.physics.wigner import wigner_discretized

r = lambda r_dB: r_dB / 20 * np.log(10)

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

##################################################################################################

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

