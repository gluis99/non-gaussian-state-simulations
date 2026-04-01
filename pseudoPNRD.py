import numpy as np
from scipy.special import genlaguerre as Laguerre
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
colwidth = 3.41667
fs = 10
mpl.rcParams.update({'font.size': fs, 
                     "text.usetex":  True, 
                     'font.family':'serif',
                     'xtick.labelsize': 8, 
                     'ytick.labelsize':8 })

fig_kwargs = {'dpi':800, 'bbox_inches': 'tight'}

# Compute Stirling numbers of the second kind S(k, n) for 0 <= k <= max_k, 0 <= n <= max_n
def stirling2(max_k: int, max_n: int) -> np.ndarray:
    """
    Precompute Stirling numbers of the second kind S(k, n)
    for 0 <= k <= max_k, 0 <= n <= max_n.
    Recurrence: S(k, n) = n * S(k-1, n) + S(k-1, n-1)
    """
    S = np.zeros((max_k + 1, max_n + 1), dtype=np.int64)
    S[0, 0] = 1
    for k in range(1, max_k + 1):
        for n in range(1, min(k, max_n) + 1):
            S[k, n] = n * S[k - 1, n] + S[k - 1, n - 1]
    return S

# Compute probability matrix P[n, k]: probability of measuring n clicks
# given k incident photons with M on/off detectors.
def prob_pseudoPNRD(M: int, k_max: int, n_max: int) -> np.ndarray:
    """
    Return matrix P[n, k]: probability of measuring n clicks
    given k incident photons with M on/off detectors.

    """
    if n_max > M:
        raise ValueError(f"n_max={n_max} cannot exceed M={M} bins")

    # Precompute full Stirling table once
    S = stirling2(k_max, n_max)

    probabilities = np.zeros((n_max + 1, k_max + 1))

    for k in range(k_max + 1):
        for n in range(min(k, n_max) + 1):          # n <= k only; rest stay zero
            if n > M:
                continue                # can't have more clicks than bins
            
            prefix = math.factorial(M) // math.factorial(M - n)  # M!/(M-n)!

            if n == k:
                probabilities[n, k] = prefix / (M ** n)
            else:                       # k > n
                probabilities[n, k] = prefix / (M ** k) * S[k, n]

    return probabilities

def pPNRD_diag(M,n):
    if n > M:
        return 0
    return math.factorial(M) // math.factorial(M - n) /M**(n)

def show_pPNRD_diag(M, n_max):
    n = np.arange(0, n_max + 1)
    P_M = np.array([pPNRD_diag(M, ni) for ni in n])
    
    plt.figure(figsize=(6, 4))
    plt.bar(n, P_M, color='skyblue', edgecolor='black')
    plt.title(f'Probability of correctly identifying n photons with M={M}')
    plt.xlabel('Photons n')
    plt.ylabel('$P(n\\,|\\,k)$')
    plt.xticks(n)
    plt.grid()
    plt.show()

# Matrix of probabilities P_M(n | k) up to some cutoff
def show_pPNRD_probabilities(M, k_max, n_max):
    P = prob_pseudoPNRD(M, k_max, n_max)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(P, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, label='$P(n\\,|\\,k)$')
    plt.xlabel('Incident photons $k$')
    plt.ylabel('Measured clicks $n$')
    plt.title(f'pseudo PNRD Probability Matrix ($M={M}$)')
    plt.xticks(np.arange(k_max + 1))
    plt.yticks(np.arange(n_max + 1))
    plt.tight_layout()
    plt.show()

# Probabilities of measuring fixed number of clicks n
def show_pPNRD_row(M, Ns, k_max):
    if isinstance(Ns, int):
        Ns = [Ns]
    n_plots = len(Ns)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    for ax, n in zip(axes[0], Ns):
        P = prob_pseudoPNRD(M, k_max, n)[n, :]
        ax.bar(np.arange(k_max + 1), P, color='lightcoral', edgecolor='black')
        ax.set_title(f'$n={n}$, $M={M}$')
        ax.set_xlabel('Incident photons $k$')
        ax.set_ylabel(f'$P(n={n}\\,|\\,k)$')
        ax.set_xticks(np.arange(k_max + 1))
        ax.grid()
    fig.suptitle(f'pseudo-PNRD: probability of measuring $n$ clicks ($M={M}$)')
    plt.tight_layout()
    plt.show()

# Probabilities given a fixed number of incident photons k
def show_pPNRD_column(M, Ks, n_max):
    if isinstance(Ks, int):
        Ks = [Ks]
    k_plots = len(Ks)
    fig, axes = plt.subplots(1, k_plots, figsize=(5 * k_plots, 4), squeeze=False)
    for ax, k in zip(axes[0], Ks):
        n = np.arange(0, n_max + 1)
        P = prob_pseudoPNRD(M, k, n_max)[:, k]
        ax.bar(n, P, color='lightgreen', edgecolor='black')
        ax.set_title(f'Probability of measuring n clicks given k={k} photons with M={M}')
        ax.set_xlabel('Measured clicks n')
        ax.set_ylabel('$P(n\\,|\\,k)$')
        ax.set_xticks(n)
        ax.grid()
    fig.suptitle(f'pseudo-PNRD: probability distribution of measured clicks $n$ for $k={k}$ photons ($M={M}$)')
    plt.tight_layout()
    plt.show()
