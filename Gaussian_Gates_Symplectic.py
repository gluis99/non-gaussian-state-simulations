import numpy as np

# Symplectic form generator of size 2 x 2
epsilon = np.array([[0, 1], [-1, 0]])
# Symplectic form generator of size 2n x 2n
def Omega(n: int):
    """
    Generate the symplectic form Omega for n modes.

    Parameters
    ----------
    n : int
        Number of modes.

    Returns
    -------
    NDArray
        Symplectic form Omega, shape (2n, 2n).
    """
    # Blockdiagonal matrix with n blocks of epsilon
    return np.kron(np.eye(n), epsilon)

# Symplectic form for two vectors x, y in R^2n
def symplectic_form(x, y):
    if len(x) != len(y) or len(x) % 2 != 0:
        raise ValueError("Input vectors must have the same even length and be in R^2n.")
    
    n = len(x) // 2
    
    if n == 1:
        return x[0]*y[1] - x[1]*y[0]
    else:
        return np.einsum('ij,jk,kl->il', x.T, Omega(n), y) 
###################################################################################
# Gaussian Transformations as symplectic matrices

#One mode Gaussian transformations
def One_Mode_Squeeze(r,theta = 0):
    """Single mode squeezing symplectic transformation"""
    S = np.array([[np.cosh(r)-np.sinh(r)*np.cos(theta), -np.sinh(r)*np.sin(theta)],
                  [-np.sinh(r)*np.sin(theta), np.cosh(r)+np.sinh(r)*np.cos(theta)]])
    return S

def Phase_rotation(theta):
    """Single mode phase rotation symplectic transformation"""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def One_Mode_Symplectic(theta, r, phi):
    """General single mode symplectic transformation"""
    return np.einsum('ij,jk,kl->il', Phase_rotation(theta), One_Mode_Squeeze(r), Phase_rotation(phi))

# Two mode Gaussian transformations

def Beam_splitter(theta):
    """Two mode beam splitter symplectic transformation"""
    tau = np.cos(theta)**2
    BS = np.array([[np.sqrt(tau), 0, np.sqrt(1-tau), 0],
                   [0, np.sqrt(tau), 0, np.sqrt(1-tau)],
                   [-np.sqrt(1-tau), 0, np.sqrt(tau), 0],
                   [0, -np.sqrt(1-tau), 0, np.sqrt(tau)]])
    return BS

def Two_Mode_Squeeze(r, theta = 0):
    """Two mode squeezing symplectic transformation"""
    S = np.array([[np.cosh(r), 0, np.sinh(r)*np.cos(theta), np.sinh(r)*np.sin(theta)],
                  [0, np.cosh(r), np.sinh(r)*np.sin(theta), -np.sinh(r)*np.cos(theta)],
                  [np.sinh(r)*np.cos(theta), np.sinh(r)*np.sin(theta), np.cosh(r), 0],
                  [np.sinh(r)*np.sin(theta), -np.sinh(r)*np.cos(theta), 0, np.cosh(r)]])
    return S

def Controlled_Z(phi):
    """Two mode controlled-Z gate symplectic transformation"""
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, phi, 0],
                   [0, 0, 1, 0],
                   [phi, 0, 0, 1]])
    return CZ

#################################################################################
# N-mode gate builders (defer N to the state)
def S(mode, r, theta=0.0):
    """Builder for N-mode single-mode squeezing on a given mode."""
    return lambda N: One_Mode_Squeeze_N_mode(r, theta, mode, N)

def R(mode, theta):
    """Builder for N-mode phase rotation on a given mode."""
    return lambda N: Phase_rotation_N_mode(theta, mode, N)

def BS(modes, theta):
    """Builder for N-mode beam splitter on a pair of modes."""
    mode1, mode2 = modes
    return lambda N: Beam_splitter_N_mode(theta, mode1, mode2, N)

def TMS(modes, r, theta=0.0):
    """Builder for N-mode two-mode squeezing on a pair of modes."""
    mode1, mode2 = modes
    return lambda N: Two_Mode_Squeeze_N_mode(r, theta, mode1, mode2, N)

def CZ(modes, phi):
    """Builder for N-mode controlled-Z on a pair of modes."""
    mode1, mode2 = modes
    return lambda N: Controlled_Z_N_mode(phi, mode1, mode2, N)

#################################################################################
#N mode Gaussian transformations
def One_Mode_Squeeze_N_mode(r, theta, mode, N):
    """N mode squeezing symplectic transformation on specified mode (0 to N-1)"""
    S_single = One_Mode_Squeeze(r, theta)
    S = np.identity(2*N)
    S[2*mode:2*mode+2, 2*mode:2*mode+2] = S_single
    return S

def Phase_rotation_N_mode(theta, mode, N):
    """N mode phase rotation symplectic transformation on specified mode (0 to N-1)"""
    R_single = Phase_rotation(theta)
    R = np.identity(2*N)
    R[2*mode:2*mode+2, 2*mode:2*mode+2] = R_single
    return R

def Beam_splitter_N_mode(theta, mode1, mode2, N):
    """N mode beam splitter symplectic transformation on specified modes (0 to N-1)"""
    BS_single = Beam_splitter(theta)
    BS = np.identity(2*N)
    BS[2*mode1:2*mode1+2, 2*mode1:2*mode1+2] = BS_single[0:2, 0:2]
    BS[2*mode1:2*mode1+2, 2*mode2:2*mode2+2] = BS_single[0:2, 2:4]
    BS[2*mode2:2*mode2+2, 2*mode1:2*mode1+2] = BS_single[2:4, 0:2]
    BS[2*mode2:2*mode2+2, 2*mode2:2*mode2+2] = BS_single[2:4, 2:4]
    return BS

def Controlled_Z_N_mode(phi, mode1, mode2, N):
    """N mode controlled-Z gate symplectic transformation on specified modes (0 to N-1)"""
    CZ_single = Controlled_Z(phi)
    CZ = np.identity(2*N)
    CZ[2*mode1:2*mode1+2, 2*mode1:2*mode1+2] = CZ_single[0:2, 0:2]
    CZ[2*mode1:2*mode1+2, 2*mode2:2*mode2+2] = CZ_single[0:2, 2:4]
    CZ[2*mode2:2*mode2+2, 2*mode1:2*mode1+2] = CZ_single[2:4, 0:2]
    CZ[2*mode2:2*mode2+2, 2*mode2:2*mode2+2] = CZ_single[2:4, 2:4]
    return CZ

def Two_Mode_Squeeze_N_mode(r, theta, mode1, mode2, N):
    """N mode two-mode squeezing symplectic transformation on specified modes (0 to N-1)"""
    S_single = Two_Mode_Squeeze(r, theta)
    S = np.identity(2*N)
    S[2*mode1:2*mode1+2, 2*mode1:2*mode1+2] = S_single[0:2, 0:2]
    S[2*mode1:2*mode1+2, 2*mode2:2*mode2+2] = S_single[0:2, 2:4]
    S[2*mode2:2*mode2+2, 2*mode1:2*mode1+2] = S_single[2:4, 0:2]
    S[2*mode2:2*mode2+2, 2*mode2:2*mode2+2] = S_single[2:4, 2:4]
    return S
