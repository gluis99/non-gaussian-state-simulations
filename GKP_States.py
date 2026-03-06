import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Gaussian_Gates_Symplectic import *


class GKPState:
    """
    A class representing a GKP (Gottesman-Kitaev-Preskill) quantum state (grid state).
    As a grid defined by scaling factor (l), a normalized generator matrix (M) and displacement vector (displacement/displacement_norm).

    Attributes
    ----------
    l : float
        Lattice scale (for computational state l = 2 *sqrt(pi))

    M : NDArray
        Generator matrix defining the GKP state. 
        (Now 2x2 for N=1, may be extended to 2Nx2N for N>1 in the future)

    displacement : NDArray
        Physical displacement vector for the GKP state.

    displacement_norm : NDArray
        Normalized displacement vector, such that displacement = l * displacement_norm.

    Maybe in the future:
    N : int
        Number of modes. For the moment, only implemented for N=1. 
        (may be extended to N>1 in the future)
    
    """
    def __init__(self, Mat: np.ndarray, displacement: np.ndarray = None, l = None, norm_disp: bool = False) -> None:
        """
        Initialize a GKP state.

        Parameters
        ----------
        Max : NDArray
            Unnormalized (unless l given) generator matrix defining the GKP state.
        displacement : NDArray, optional
            Displacement vector for the GKP state.
        norm_disp : bool, optional
            If True, `displacement` is interpreted as normalized coordinates.
            If False, `displacement` is interpreted as physical coordinates and
            converted via `displacement_norm = displacement / l`.
        """
        Omega = epsilon # Symplectic form for N=1

        if l == None:
            # Lattice scaling
            l = np.abs(np.linalg.det(Mat))
            self.M = Mat /l
        else:
            # Need to implement way to assure that M is normalized
            self.M = Mat
        self.l = l
        if self.l == 0:
            raise ValueError("l must be non-zero.")
        if displacement is None:
            displacement = np.zeros(2)
        
        displacement = np.asarray(displacement, dtype=float)
        if norm_disp:
            self.displacement_norm = displacement
        else:
            self.displacement_norm = displacement / self.l

    # Physical displacement from normalized
    @property
    def displacement(self) -> np.ndarray:
        """
        Physical displacement vector (same units as x and p).
        """
        return self.l * self.displacement_norm

    # Normalized displacement from physical
    @displacement.setter
    def displacement(self, value: np.ndarray) -> None:
        """
        Set displacement from physical coordinates.
        """
        value = np.asarray(value, dtype=float)
        self.displacement_norm = value / self.l

    # Not sure if still correct after changes
    @classmethod
    def canonical_GKP(cls) -> "GKPState":
        """
        Create a canonical GKP state.

        Returns
        -------
        GKPState
            Canonical GKP state with generator matrix [[1, 0], [0, 1]].
        """
        M = np.array([[0, 1], [1, 0]])
        l = np.sqrt(2*np.pi)
        displacement = np.zeros(2)
        return cls(M, displacement, l)
    
    @classmethod
    def computational_GKP(cls, b, l = 2 *np.sqrt(np.pi)) -> "GKPState":
        """
        Create a (scaled) computational GKP state.

        Parameters
        ----------
        b : int
            Binary value (0 or 1) for the computational GKP state.

        Returns
        -------
        GKPState
            Computational GKP qubit state |b>_L.
        """
        M = np.array([[0, 1], [1, 0]])
        if b == 0:
            displacement = np.zeros(2)
            norm_disp = True
        elif b == 1:
            # Displaced by action of X_L = sqrt(S_2)
            displacement = np.array([0.5, 0.0])
            norm_disp = True
        else:
            raise ValueError("b must be 0 or 1.")
        return cls(M, displacement, l, norm_disp=norm_disp)
    
    def gram_matrix(self) -> np.ndarray:
        """
        Compute the Gram matrix (A) of the GKP state.

        Returns
        -------
        NDArray
            Gram matrix A = M^T @ Omega @ M, which should have integer entries for a valid GKP state.
        """
        #if self.N == 1:
        if True:
            Omega = epsilon
        else:
            Omega = Omega(self.N)
        A = np.einsum('ij,jk,kl->il', self.M.T, Omega, self.M)
        return A
    
    def dimension(self) -> int:
        """
        Compute the dimension of the logical subspace encoded by the GKP state.

        Returns
        -------
        int
            Dimension of the logical subspace, given by sqrt(det(A)) where A is the Gram matrix.
        """
        A = self.gram_matrix()

        scale_factor = self.l**2/(2*np.pi)
        return int(np.round(np.sqrt(np.linalg.det(A)) * scale_factor))
    
    def stabilizers(self, print_stabilizers=False):
        """
        Compute the stabilizers of the GKP state.

        Returns
        -------
        tuple of callables
            Stabilizer functions S_1 and S_2 corresponding to the grid vectors defined by M.
        """
        stabilizers = []
        for v in self.M.T:
            if print_stabilizers:
                print(f'exp(i * {self.l} * (x*({v[1]}) - p*({v[0]}) + phi))')
                # Extra phase irrelevant for stabilizers
                print(f'phi = {symplectic_form(self.displacement, v)}')
            stabilizers.append(lambda x,p: np.exp(1j *self.l * symplectic_form(np.array([x,p]), v)))
        return tuple(stabilizers)

    # Only if states equivalent to computational GKP states
    def logical_operators(self, print_logicals = False):
        """
        Compute the logical operators of the GKP state.

        Returns
        -------
        tuple of callables
            Logical operator functions X_L and Z_L corresponding to the grid vectors defined by M.
        """
        logical_ops = []
        for v in self.M.T:
            logical_ops.append(lambda x,p: np.exp(0.5j *self.l * symplectic_form(np.array([x,p]), v)))
        if print_logicals:
            print(f'X_L = exp(i * {self.l/2} * (x*({self.M[1,1]}) - p*({self.M[0,1]}) + phi))')
            print(f'Z_L = exp(i * {self.l/2} * (x*({self.M[1,0]}) - p*({self.M[0,0]}) + phi))')

            # Extra phases irrelevant for logical operators
            print(f'phi_X = {symplectic_form(self.displacement, self.M[:,1])}')
            print(f'phi_Z = {symplectic_form(self.displacement, self.M[:,0])}')
        return tuple(logical_ops)
    
    def apply_gaussian_gate(self, S: np.ndarray, d = np.array([0,0])) -> "GKPState":
        """
        Apply a linear (symplectic) Gaussian gate in-place.

        Parameters
        ----------
        S : NDArray or callable
            Symplectic matrix (2N x 2N) or a callable that builds it from N.
        d : NDArray, shape (2N,), optional
            Displacement vector. Defaults to zero.

        Returns
        -------
        GKPState
            Self after transformation.
        """
        #if self.N == 1:
        if True:
            Sinv = - np.einsum('ij,jk,kl->il', epsilon, S.T, epsilon)
        else:
            Sinv = - np.einsum('ij,jk,kl->il', Omega(self.N), S.T, Omega(self.N))

        self.M = np.einsum('ij,jk->ik', Sinv, self.M)
        self.displacement = np.einsum('ij,j->i', Sinv, self.displacement) + d
        self.displacement_norm = self.displacement / self.l

        return self

    def expectation_value(
        self,
        O,
        x_range=(-5, 5),
        p_range=(-5, 5),
        num_points=801,
        delta_x=0.2,
        delta_p=None,
    ):
        """
        Expectation value of observable O(x,p)
        with respect to this GKP state's Wigner function.
        """

        W, X, P = self.wigner_approx(
            x_range=x_range,
            p_range=p_range,
            num_points=num_points,
            delta_x=delta_x,
            delta_p=delta_p,
            normalize=True,
        )

        O_values = O(X, P)

        dx = X[0, 1] - X[0, 0]
        dp = P[1, 0] - P[0, 0]

        return np.sum(W * O_values) * dx * dp

    #################################################################################
    # Hermitian operator minimized only by GKP state
    # See Marek, PRL 132, 210601 (2024), DOI: 10.1103/PhysRevLett.132.210601.
    def Q_operator(self, scaling=None, print_Q = False):
        """
        Compute (and print) the Q operator for the GKP state at given x and p.

        Parameters
        ----------
        x : float
            Position variable.
        p : float
            Momentum variable.

        print_Q : bool, optional
            If True, print the symbolic form of the Q operator.

        Returns
        -------
        float
            Value of the Q operator at (x, p).
        """
        dim = self.dimension()
        v_1, v_2 = self.M.T
        disp = self.displacement_norm
        
        if scaling is None:
            scaling = 1

        if print_Q:
                print(f'Grid with: scaling s={scaling}, l={self.l}, v1=({v_1[0]}, {v_1[1]})^T, v2=({v_2[0]}, {v_2[1]})^T, displacement=({disp[0]}, {disp[1]})')
                print(f'Q^{{(s)}}_{{l,v_1,v_2,d}} = 2 - cos(1/2 *{scaling}*{self.l}*[x*({v_1[1]}) - p*({v_1[0]}) +phi_1]) - cos( {scaling}*{self.l}*[x*({v_2[1]}) - p*({v_2[0]}) + phi_2])')
                print(f'phi1 = {symplectic_form(disp, v_1)}')
                print(f'phi2 = {symplectic_form(disp, v_2)}')
        
        return lambda x,p: 2 - np.cos(0.5*scaling*self.l *(symplectic_form(np.array([x,p]), v_1) + symplectic_form(disp, v_1) )) \
                             - np.cos(    scaling*self.l *(symplectic_form(np.array([x,p]), v_2) + symplectic_form(disp, v_2) ))

    # Approximation of the expectation value of the Q operator
    # xi = integral dx dp W(x, p) Q(x, p)
    @staticmethod
    def _grid_spacing_from_values(values):
        """
        Infer grid spacing from a 1D coordinate array.
        """
        values = np.asarray(values, dtype=float).ravel()
        values = np.unique(values)
        if values.size < 2:
            raise ValueError("Grid must contain at least two distinct points.")

        diffs = np.diff(values)
        diffs = diffs[np.abs(diffs) > 0]
        if diffs.size == 0:
            raise ValueError("Could not infer non-zero grid spacing from mesh.")

        return float(np.mean(np.abs(diffs)))

    def _normalize_wigner_plot(self, wigner_plot):
        """
        Normalize Wigner input to (W, X, P, dx, dp), supporting:
        - (W, X_mesh, P_mesh)
        - (W, x_axis, p_axis)
        """
        if len(wigner_plot) != 3:
            raise ValueError("wigner_plot must be a tuple (W, X, P).")

        W, X_raw, P_raw = wigner_plot
        W = np.asarray(W, dtype=float)
        X_raw = np.asarray(X_raw, dtype=float)
        P_raw = np.asarray(P_raw, dtype=float)

        if W.ndim != 2:
            raise ValueError("W must be a 2D array.")

        if X_raw.ndim == 2 and P_raw.ndim == 2:
            if X_raw.shape != W.shape or P_raw.shape != W.shape:
                raise ValueError("When X and P are 2D, they must match W shape.")
            X, P = X_raw, P_raw
            dx = self._grid_spacing_from_values(X)
            dp = self._grid_spacing_from_values(P)
            return W, X, P, dx, dp

        if X_raw.ndim == 1 and P_raw.ndim == 1:
            x_axis = X_raw
            p_axis = P_raw

            if W.shape == (p_axis.size, x_axis.size):
                X, P = np.meshgrid(x_axis, p_axis, indexing="xy")
            elif W.shape == (x_axis.size, p_axis.size):
                X, P = np.meshgrid(x_axis, p_axis, indexing="ij")
            else:
                raise ValueError(
                    "For 1D X and P, W shape must be either (len(P), len(X)) "
                    "or (len(X), len(P))."
                )

            dx = self._grid_spacing_from_values(x_axis)
            dp = self._grid_spacing_from_values(p_axis)
            return W, X, P, dx, dp

        raise ValueError(
            "X and P must both be 2D meshes or both be 1D axis arrays."
        )

    def xi_approx(self, wigner_plot, scaling=None):
        W, X, P, dx, dp = self._normalize_wigner_plot(wigner_plot)

        Q_values = self.Q_operator(scaling=scaling)(X, P)

        return float(np.real_if_close(np.sum(W * Q_values) * dx * dp, tol=1e6))

    ##################################################################################
    # Pipeline for scaled-grid comparison analysis using xi(s)

    def precompute_phases(self, wigner_plot):
        """
        Precompute phases of cosines appearing in Q operator.
        This removes repeated expensive evaluations during scaling scan.
        """
        W, X, P, _, _ = self._normalize_wigner_plot(wigner_plot)
        v1, v2 = self.M.T
        disp = self.displacement_norm
        l = self.l

        phi1 = l*(symplectic_form(np.array([X, P]), v1)
                            + symplectic_form(disp, v1))

        phi2 = l*(symplectic_form(np.array([X, P]), v2)
                            + symplectic_form(disp, v2))

        return phi1, phi2

    def xi_from_phases(self, W, dx, dp, phi1, phi2, s, normalized=True):
        """
        Compute xi(s) or xi(s)/sqrt(s) using precomputed phases.
        """
        Q = 2 - np.cos(0.5 * s * phi1) - np.cos(s * phi2)
        xi = np.sum(W * Q) * dx * dp

        if normalized:
            if s == 0:
                return np.inf
            xi = xi / (np.sqrt(s))

        return float(np.real_if_close(xi, tol=1e6))

    # Taken from AI-generated code, to be adapted
    @staticmethod
    def _strict_local_minima_indices(values):
        """
        Strict local minima in 1D array.
        """
        values = np.asarray(values, dtype=float)
        if values.size < 3:
            return np.array([], dtype=int)
        return np.where(
            (values[1:-1] < values[:-2]) &
            (values[1:-1] < values[2:])
        )[0] + 1

    def scan_scaling(
        self,
        wigner_plot,
        s_min=0.2,
        s_max=4.0,
        num_points=301,
        normalized_xi=True,
    ):
        """
        Scan xi(s) or xi(s)/s^2 over scaling parameter s.
        """
        if s_max <= s_min:
            raise ValueError("s_max must be > s_min")

        W, X, P, dx, dp = self._normalize_wigner_plot(wigner_plot)
        phi1, phi2 = self.precompute_phases((W, X, P))

        # s is scanned directly
        scales = np.linspace(s_min, s_max, num_points)
        xis = np.array([
            self.xi_from_phases(W, dx, dp, phi1, phi2, s, normalized_xi)
            for s in scales
        ])

        local_idx = self._strict_local_minima_indices(xis)
        if len(local_idx) == 0:
            best_idx = int(np.argmin(xis))
        else:
            best_idx = int(local_idx[np.argmin(xis[local_idx])])

        return {
            "scales": scales,
            "xis": xis,
            "best_index": best_idx,
            "best_scale": float(scales[best_idx]),  # this is s
            "best_xi": float(xis[best_idx]),
            "local_indices": local_idx,
        }

    def scaled(self, s):
        """
        Return a copy with l_new = l * s.
        """
        s = float(s)
        if s <= 0:
            raise ValueError("s must be positive.")

        return GKPState(
            Mat=self.M.copy(),
            displacement=self.displacement_norm.copy(),
            l=self.l * s,
            norm_disp=True,
        )


    def compare_scaled_grid(
        self,
        wigner_plot,
        s_min=0.2,
        s_max=4.0,
        num_points=301,
        normalized_xi=False,
        plot_scan=True,
        plot_best_grid=True,
        plot_all_candidates=True,
        max_candidates=None,
        delta_fixed=0.3,
    ):
        fit = self.scan_scaling(
            wigner_plot,
            s_min=s_min,
            s_max=s_max,
            num_points=num_points,
            normalized_xi=normalized_xi,
        )

        scales = fit["scales"]            # s values
        xis = fit["xis"]
        best_scale = fit["best_scale"]    # best s
        best_xi = fit["best_xi"]
        local_idx = fit["local_indices"]

        if len(local_idx) == 0:
            candidate_idx = np.array([fit["best_index"]], dtype=int)
        else:
            candidate_idx = np.asarray(local_idx, dtype=int)

        if max_candidates is not None:
            max_candidates = int(max_candidates)
            if max_candidates <= 0:
                raise ValueError("max_candidates must be a positive integer or None.")
            order = np.argsort(xis[candidate_idx])
            candidate_idx = candidate_idx[order[:max_candidates]]

        fit["candidate_indices"] = candidate_idx
        fit["candidate_scales"] = scales[candidate_idx]
        fit["candidate_xis"] = xis[candidate_idx]

        print(f"Best scaling s: {best_scale:.6f}")
        print(f"Best xi:  {best_xi:.3f}")
        print("Candidate minima:")
        for rank, idx in enumerate(candidate_idx, 1):
            print(f"  {rank:2d}) s={scales[idx]:.6f}, xi={xis[idx]:.3f}")

        if plot_scan:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
            ax.plot(scales, xis, lw=2.2, color="tab:blue", label="scan")
            if len(local_idx) > 0:
                ax.scatter(scales[local_idx], xis[local_idx], s=42, color="tab:orange", zorder=3, label="local minima")
            ax.scatter([best_scale], [best_xi], s=70, color="crimson", edgecolors="black", linewidths=0.4, zorder=4, label="selected minimum")
            ax.set_xlabel(r"scaling $s$", fontsize=11)
            ax.set_ylabel(r"$\tilde{\xi}(s)$" if normalized_xi else r"$\xi(s)$", fontsize=11)
            ax.set_title(r"Scaling parameter scan", fontsize=12)
            ax.grid(alpha=0.28, linestyle="--", linewidth=0.6)
            ax.tick_params(labelsize=10)
            ax.legend(frameon=False, fontsize=9)
            plt.show()

        if plot_best_grid:
            W_input, X_input, P_input, _, _ = self._normalize_wigner_plot(wigner_plot)
            x_range = (X_input.min(), X_input.max())
            p_range = (P_input.min(), P_input.max())
            npts = max(W_input.shape)

            plot_indices = candidate_idx if plot_all_candidates else np.array([fit["best_index"]], dtype=int)

            for idx in plot_indices:
                s_val = scales[idx]
                gkp_ref = self.scaled(s_val)  # <-- use s directly

                W_ref, X_ref, P_ref = gkp_ref.wigner_finite_energy(
                    x_range=x_range,
                    p_range=p_range,
                    num_points=npts,
                    delta_x=delta_fixed,
                    normalize=True,
                )

                vmax = np.percentile(np.abs(np.concatenate([W_input.ravel(), W_ref.ravel()])), 99.5)
                vmin = -vmax

                fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

                im0 = axes[0].imshow(W_input, extent=(x_range[0], x_range[1], p_range[0], p_range[1]), origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
                
                axes[0].set_title(r"Input Wigner $W_{\mathrm{in}}(x,p)$")
                axes[0].set_xlabel(r"$x$")
                axes[0].set_ylabel(r"$p$")
                axes[0].set_aspect("equal")

                im1 = axes[1].imshow(W_ref, extent=(x_range[0], x_range[1], p_range[0], p_range[1]), origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axes[1].set_title(r"Reference Wigner $W_{\mathrm{GKP}}(x,p)$")
                # Put fit parameters inside the right panel to keep title baselines aligned.
                axes[1].text(
                    0.02,
                    0.98,
                    rf"$s={s_val:.3f},\;\xi={xis[idx]:.3f}$",
                    transform=axes[1].transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
                axes[1].set_xlabel(r"$x$")
                axes[1].set_ylabel(r"$p$")
                axes[1].set_aspect("equal")

                cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
                cbar.set_label("Wigner value", fontsize=10)
                plt.show()

        return fit
    
    ##################################################################################
    # Pipeline for squeezed, scaled-grid comparison analysis using xi(r, s)

    ##################################################################################
    # Lattice points of the GKP state
    def _lattice_points_in_window(
        self,
        x_min: float,
        x_max: float,
        p_min: float,
        p_max: float,
        displacement=None,
        all_points: bool = False,
    ) -> np.ndarray:
        """
        Compute lattice points x = (l*M) @ n + d inside [x_min,x_max]x[p_min,p_max].
        all_points=False -> n in Z^2
        all_points=True  -> n in (1/2 Z)^2
        """
        d = np.zeros(2, dtype=float) if displacement is None else np.asarray(displacement, dtype=float).reshape(2)

        M = np.asarray(self.M, dtype=float)
        if M.shape != (2, 2):
            raise ValueError(f"Expected self.M shape (2,2), got {M.shape}")

        B = float(self.l) * M
        if abs(np.linalg.det(B)) < 1e-14:
            raise ValueError("l*M is singular; cannot generate lattice points.")

        corners = np.array(
            [[x_min, p_min], [x_min, p_max], [x_max, p_min], [x_max, p_max]],
            dtype=float,
        )

        Binv = np.linalg.inv(B)
        n_corners = (corners - d) @ Binv.T

        step = 0.5 if all_points else 1.0
        k1_min = int(np.floor(np.min(n_corners[:, 0]) / step)) - 1
        k1_max = int(np.ceil(np.max(n_corners[:, 0]) / step)) + 1
        k2_min = int(np.floor(np.min(n_corners[:, 1]) / step)) - 1
        k2_max = int(np.ceil(np.max(n_corners[:, 1]) / step)) + 1

        c1 = np.arange(k1_min, k1_max + 1) * step
        c2 = np.arange(k2_min, k2_max + 1) * step
        cc1, cc2 = np.meshgrid(c1, c2, indexing="ij")
        coeffs = np.column_stack((cc1.ravel(), cc2.ravel()))

        pts = coeffs @ B.T + d
        mask = (
            (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
            (pts[:, 1] >= p_min) & (pts[:, 1] <= p_max)
        )
        return pts[mask]

    def show_lattice_points(
        self,
        wigner_plot,
        displacement_from_0=None,
        all_points: bool = False,
        show_plot: bool = False,
    ):
        """
        Plot any provided Wigner grid and overlay GKP lattice peaks.
        all_points=False -> primitive repeating structure
        all_points=True  -> all lattice points included
        """
        W, X, P, _, _ = self._normalize_wigner_plot(wigner_plot)

        x_axis = np.unique(X)
        p_axis = np.unique(P)

        if W.shape == (p_axis.size, x_axis.size):
            W_plot = W
        elif W.shape == (x_axis.size, p_axis.size):
            W_plot = W.T
        else:
            raise ValueError(f"Incompatible shapes: W{W.shape}, X{X.shape}, P{P.shape}")

        x_min, x_max = float(x_axis.min()), float(x_axis.max())
        p_min, p_max = float(p_axis.min()), float(p_axis.max())

        pts = self._lattice_points_in_window(
            x_min=x_min,
            x_max=x_max,
            p_min=p_min,
            p_max=p_max,
            displacement=displacement_from_0,
            all_points=all_points,
        )

        # Symmetric color scale around 0 -> white at 0
        vmax = float(np.max(np.abs(W_plot)))
        if vmax == 0:
            vmax = 1e-12
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            W_plot,
            extent=(x_min, x_max, p_min, p_max),
            origin="lower",
            cmap="RdBu_r",
            norm=norm,
            interpolation="none",
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, label=r"$W(x,p)$")
        ax.scatter(pts[:, 0], pts[:, 1], s=20, c="k", marker="o", label="Lattice points")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$p$")
        ax.set_title(r"Wigner distribution with GKP lattice points")
        ax.legend(loc="upper right")
        fig.tight_layout()

        if show_plot:
            plt.show()

        return fig, ax

    ##################################################################################

    # Wigner function related
    def _resolve_finite_energy_widths(self, delta_x, delta_p):
        """
        Resolve finite-energy widths for approximate GKP states.
        """
        if delta_x is None and delta_p is None:
            delta_x = 0.2
            delta_p = 0.2
        elif delta_x is None:
            delta_x = delta_p
        elif delta_p is None:
            delta_p = delta_x

        if delta_x <= 0 or delta_p <= 0:
            raise ValueError("delta_x and delta_p must be positive.")

        return float(delta_x), float(delta_p)
    
    # Wigner function of approximate GKP state, taken from AI-generated code, to be adapted
    def wigner_finite_energy(
        self,
        x_range=(-5, 5),
        p_range=(-5, 5),
        num_points=801,
        delta_x=0.2,
        delta_p=None,
        envelope_cutoff=4.0,
        peak_cutoff=4.0,
        normalize=True,
    ):
        """
        Approximate finite-energy GKP Wigner function.

        Uses Gaussian peaks with widths `delta_x` and `delta_p` in x and p,
        and an inverse-width envelope consistent with finite-energy modeling:
        envelope ~ exp(-0.5 * ((delta_p*x0)^2 + (delta_x*p0)^2)).

        Lattice points are generated using an `l * M` Bravais lattice plus a
        four-point half-cell basis:

            point = n @ (l*M)^T + (r/2) @ (l*M)^T + displacement,

        with r in {0,1}^2 and basis sign (-1)^(r1*r2). This preserves
        stabilizer translations by l while retaining nontrivial logical
        half-translations.

        """

        delta_x, delta_p = self._resolve_finite_energy_widths(
            delta_x=delta_x,
            delta_p=delta_p,
        )

        x = np.linspace(x_range[0], x_range[1], num_points)
        p = np.linspace(p_range[0], p_range[1], num_points)
        X, P = np.meshgrid(x, p)

        dx = x[1] - x[0]
        dp = p[1] - p[0]

        W = np.zeros_like(X)

        # Determine which lattice points could contribute

        # Bounding rectangle from envelope (inverse-width finite-energy envelope)
        envelope_x_bound = envelope_cutoff / delta_p
        envelope_p_bound = envelope_cutoff / delta_x

        # Bounding rectangle from plotting window + peak widths
        window_x_bound = max(abs(x_range[0]), abs(x_range[1])) + peak_cutoff * delta_x
        window_p_bound = max(abs(p_range[0]), abs(p_range[1])) + peak_cutoff * delta_p

        bound_x = max(envelope_x_bound, window_x_bound)
        bound_p = max(envelope_p_bound, window_p_bound)

        effective_M = self.l * self.M

        # Half-cell offsets can extend points from Bravais-node centers.
        cell_half_span = 0.5 * np.max(np.linalg.norm(effective_M.T, axis=1))
        max_radius = np.sqrt(bound_x**2 + bound_p**2) + cell_half_span

        # Convert radius to lattice index radius

        singular_values = np.linalg.svd(effective_M, compute_uv=False)
        s_min = max(np.min(singular_values), 1e-12)

        lattice_radius = int(np.ceil(max_radius / s_min))

        k_vals = np.arange(-lattice_radius, lattice_radius + 1)
        basis_indices = ((0, 0), (1, 0), (0, 1), (1, 1))

        for k1 in k_vals:
            for k2 in k_vals:

                n = np.array([k1, k2], dtype=float)

                for r1, r2 in basis_indices:
                    r = np.array([r1, r2], dtype=float)
                    point = n @ effective_M.T + 0.5 * (r @ effective_M.T) + self.displacement

                    # Envelope cutoff pruning
                    envelope_argument = (delta_p * point[0])**2 + (delta_x * point[1])**2
                    if envelope_argument > envelope_cutoff**2:
                        continue

                    # Window pruning
                    if (
                        point[0] < x_range[0] - peak_cutoff * delta_x
                        or point[0] > x_range[1] + peak_cutoff * delta_x
                        or point[1] < p_range[0] - peak_cutoff * delta_p
                        or point[1] > p_range[1] + peak_cutoff * delta_p
                    ):
                        continue

                    parity = (-1.0) ** (r1 * r2)
                    envelope = np.exp(-0.5 * envelope_argument)

                    W += (
                        parity
                        * envelope
                        * np.exp(
                            -0.5 * (
                                ((X - point[0]) / delta_x)**2
                                + ((P - point[1]) / delta_p)**2
                            )
                        )
                    )

        W *= 1.0 / (2 * np.pi * delta_x * delta_p)

        if normalize:
            norm = np.sum(W) * dx * dp
            if norm != 0:
                W /= norm

        return W, X, P

    def plot_wigner_finite_energy(
        self,
        x_range=(-5, 5),
        p_range=(-5, 5),
        num_points=801,
        delta_x=0.2,
        delta_p=None,
        scale_axes_by_l=True,
        color_scale="robust",
        color_percentile=99.5,
        symmetric_color=True,
    ):
        """
        Plot the canonical finite-energy GKP Wigner function
        using imshow (fast and clean).

        Parameters
        ----------
        x_range, p_range : tuple
            Plot window.
        num_points : int
            Grid resolution per axis.
        delta_x, delta_p : float, optional
            Peak widths in x and p. If only one is given, symmetric widths are used.
        scale_axes_by_l : bool
            If True, axes shown in units of l.
        color_scale : {"robust", "maxabs"}
            Strategy used to set color limits.
            - "robust": uses a high percentile of |W| for better contrast.
            - "maxabs": uses the absolute maximum (previous behavior).
        color_percentile : float
            Percentile used when color_scale="robust".
        symmetric_color : bool
            If True, uses symmetric limits [-v, v].
        """

        # Get normalized canonical Wigner function
        W, X, P = self.wigner_finite_energy(
            x_range=x_range,
            p_range=p_range,
            num_points=num_points,
            delta_x=delta_x,
            delta_p=delta_p,
            normalize=True,
        )

        # Adaptive color scale
        absW = np.abs(W)
        if color_scale == "robust":
            vmax = np.percentile(absW, color_percentile)
        elif color_scale == "maxabs":
            vmax = np.max(absW)
        else:
            raise ValueError("color_scale must be 'robust' or 'maxabs'.")

        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1e-12

        if symmetric_color:
            vmin = -vmax
        else:
            vmin = np.min(W)
            if not np.isfinite(vmin):
                vmin = -vmax

        # Axis scaling
        if scale_axes_by_l:
            unit = self.l
            extent = [
                x_range[0] / unit,
                x_range[1] / unit,
                p_range[0] / unit,
                p_range[1] / unit,
            ]
            xlabel = r"$x/\ell$"
            ylabel = r"$p/\ell$"
        else:
            extent = [
                x_range[0],
                x_range[1],
                p_range[0],
                p_range[1],
            ]
            xlabel = r"$x$"
            ylabel = r"$p$"

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

        im = ax.imshow(
            W,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Wigner value")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(r"Finite-energy GKP Wigner distribution $W_{\mathrm{GKP}}(x,p)$")

        plt.tight_layout()
        plt.show()

