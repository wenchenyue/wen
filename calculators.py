

import jax
import jax.numpy as jnp
from TB_model import TB_model
from utility import fermi_dirac_vamp, fermi_dirac_derivative_vamp
from Hk import make_k_hamiltonian_calculator, make_k_hamiltonian_velocity_calculator

# using double precision
jax.config.update("jax_enable_x64", True)
DEFAULT_TYPE_COMPLEX = jnp.complex128
DEFAULT_TYPE_REAL = jnp.float64

# Calculators==================Begin
# (E_k, Vec_k, velocity_k)-dependent calculators

# one operator
# ==============================================================================
# CALCULATORS MODULE
# ==============================================================================
# This module provides various calculators for tight-binding model computations.
# All calculators in this section are (E_k, Vec_k, velocity_k)-dependent,
# meaning they require eigenvalues, eigenvectors, and velocity operators as input.
# ==============================================================================

# Levi-Civita symbol for cross product operations
# levi_civita[gamma] = (alpha, beta) where gamma is the cross product direction
levi_civita = jnp.array([
    [1,2],  # gamma=0 (x): cross product in y-z plane
    [2,0],  # gamma=1 (y): cross product in z-x plane
    [0,1]   # gamma=2 (z): cross product in x-y plane
])

# ==============================================================================
# (E_k, Vec_k, velocity_k)-DEPENDENT CALCULATORS
# ==============================================================================

def make_calc_inner(model: TB_model, Operator: jnp.ndarray):
    """
    Create a calculator for computing the inner product of an operator with wavefunctions.
    
    This calculator computes <ψ_n(k)|O|ψ_n(k)> for each band n, where O is the given operator.
    The result represents the expectation value of the operator in each eigenstate.
    
    Common use cases:
    - If O is the identity matrix: computes occupation numbers (cumulative DOS after Fermi scan)
    - If O is a diagonal matrix with 1s for specific orbitals: computes partial DOS (PDOS)
    - If O is any Hermitian operator: computes the expectation value of that observable
    
    Args:
        model (TB_model): The tight-binding model class instance containing model parameters.
        Operator (jnp.ndarray): The operator matrix in the orbital basis, shape (nwan, nwan).
                               Must be a square matrix matching the number of Wannier functions.
    
    Returns:
        function: A JIT-compiled calculator function with signature:
                  (E_k, Vec_k, velocity_k) -> inner_product
                  where:
                    - E_k (jnp.ndarray): Eigenvalues, shape (nwan,)
                    - Vec_k (jnp.ndarray): Eigenvectors, shape (nwan, nwan)
                    - velocity_k (jnp.ndarray): Velocity operator, shape (nwan, nwan, 3)
                  Returns:
                    inner_product (jnp.ndarray): Real expectation values, shape (nwan,)
    
    Raises:
        AssertionError: If Operator shape does not match (nwan, nwan).
    
    Notes:
        - The velocity_k parameter is ignored in this calculation but kept for interface consistency.
        - The result is always real because we're computing expectation values of Hermitian operators.
        - After Fermi-Dirac smearing, states below EF contribute 1, above EF contribute 0.
    
    Example:
        >>> # Compute cumulative DOS (identity operator)
        >>> model = read_TB_model("TB.npz")
        >>> calc_dos = make_calc_inner(model, jnp.eye(model.nwan))
        >>> results = calc_dos(E_k, Vec_k, velocity_k)
    """
    nwan = model.nwan
    assert Operator.shape == (nwan,nwan), f"Operator shape should be ({nwan},{nwan}), but got {Operator.shape}"

    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        """
        Calculate the inner product <ψ_n(k)|O|ψ_n(k)> for each band n.
        
        Args:
            E_k (jnp.ndarray): Eigenvalues of H(k), shape (nwan,). Ignored in this calculation.
            Vec_k (jnp.ndarray): Eigenvectors of H(k), shape (nwan, nwan). Columns are eigenvectors.
            velocity_k (jnp.ndarray): Velocity operator, shape (nwan, nwan, 3). Ignored in this calculation.
        
        Returns:
            jnp.ndarray: Real expectation values for each band, shape (nwan,).
        """
        # Compute <ψ_n|O|ψ_n> = Σ_{i,j} ψ_n*[i] * O[i,j] * ψ_n[j]
        # Using einsum for efficient computation: 'ij,jk,ki->i' means:
        #   Vec_k.conj().T[i,j] = ψ_n*[j] (transpose of conjugate)
        #   Operator[j,k] = O[j,k]
        #   Vec_k[k,i] = ψ_n[i]
        #   Result[i] = Σ_{j,k} ψ_n*[j] * O[j,k] * ψ_n[k] = <ψ_n|O|ψ_n>
        inner_product = jnp.einsum('ij,jk,ki->i', Vec_k.conj().T, Operator, Vec_k, optimize='optimal')
        return inner_product.real
    return _calculator




# modern theory of orbital magnetism
def make_calc_mag_modern(model: TB_model, gamma: int = 2):
    """
    Create a calculator for the intrinsic orbital magnetization using modern theory.
    
    This calculator implements the modern theory of orbital magnetism based on Pezo et al.
    [PHYSICAL REVIEW B 106, 104414 (2022)], which expresses the orbital magnetic moment
    in terms of the Berry curvature and velocity operators.
    
    The orbital magnetic moment for band n at k-point k in direction gamma is:
    M_orb(n,k,gamma) = Im[0.5 * Σ_{q≠n} (1/(E_qk - E_nk)) * 
                        (<n|v_α|q><q|v_β|n> - <n|v_β|q><q|v_α|n>)]
    
    This is equivalent to the diagonal elements of the orbital magnetization matrix
    computed by make_calc_mag_matrix().
    
    Args:
        model (TB_model): The tight-binding model class instance containing model parameters.
        gamma (int, optional): Direction of magnetization (0=x, 1=y, 2=z). Default is 2 (z).
    
    Returns:
        function: A JIT-compiled calculator function with signature:
                  (E_k, Vec_k, velocity_k) -> magnetization
                  where:
                    - E_k (jnp.ndarray): Eigenvalues, shape (nwan,)
                    - Vec_k (jnp.ndarray): Eigenvectors, shape (nwan, nwan)
                    - velocity_k (jnp.ndarray): Velocity operator, shape (nwan, nwan, 3)
                  Returns:
                    magnetization (jnp.ndarray): Orbital magnetic moments, shape (nwan,)
    
    Notes:
        - Only the intrinsic part of the orbital magnetization is calculated.
        - The velocity operator may include or exclude gauge potential (A) terms, which affects results.
        - A small imaginary part (1e-12) is added to energy denominators to avoid division by zero.
        - The result is purely imaginary, so we take the imaginary part to get the physical quantity.
    
    References:
        Pezo et al., PHYSICAL REVIEW B 106, 104414 (2022), Eq. (6)
    
    Example:
        >>> # Calculate z-component of orbital magnetization
        >>> model = read_TB_model("TB.npz")
        >>> calc_mag = make_calc_mag_modern(model, gamma=2)
        >>> mag_moments = calc_mag(E_k, Vec_k, velocity_k)
    """
    nwan = model.nwan
    print(f"modern theory of orbital magnetism, caution: only calculate the intrinsic part")
    alpha, beta = levi_civita[gamma]
    
    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        """
        Calculate the orbital magnetization for each band at a given k-point.
        
        Args:
            E_k (jnp.ndarray): Eigenvalues of H(k), shape (nwan,).
            Vec_k (jnp.ndarray): Eigenvectors of H(k), shape (nwan, nwan). Columns are eigenvectors.
            velocity_k (jnp.ndarray): Velocity operator in orbital basis, shape (nwan, nwan, 3).
        
        Returns:
            jnp.ndarray: Orbital magnetic moments for each band, shape (nwan,).
        """
        Vec_k_dagger = Vec_k.conj().T
        
        # Transform velocity operators to band basis: v_band = U^† * v_orb * U
        v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        v_beta = Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k
        
        # Energy differences: Ediff[q, n] = E_q - E_n
        Ediff = E_k[:,None] - E_k[None,:]
        
        # Green's function: G[q, n] = 1/(E_q - E_n)
        # Small imaginary part (1e-12j) avoids division by zero
        Gqn = 1 / (Ediff + 1e-12*1j)
        
        # Set diagonal elements to 0 to exclude q=n terms (no intraband contribution)
        Gqn = jnp.fill_diagonal(Gqn, 0.0, inplace=False)
        
        # Compute cross product terms in direction gamma: v_α × v_β
        # term1[n] = Σ_q G[q,n] * v_α[n,q] * v_β[q,n]
        cross_term1 = jnp.einsum("qn,nq,qn->n", Gqn, v_alpha, v_beta, optimize='optimal')
        
        # term2[n] = Σ_q G[q,n] * v_β[n,q] * v_α[q,n]
        cross_term2 = jnp.einsum("qn,nq,qn->n", Gqn, v_beta, v_alpha, optimize='optimal')
        
        # Orbital magnetization: M = 0.5 * Im(cross_term1 - cross_term2)
        # Note: cross_term2 = cross_term1* (complex conjugate)
        # So M = Im(cross_term1) is equivalent
        magnetism = jnp.imag((cross_term1 - cross_term2) / 2)
        return magnetism 
    return _calculator


def make_calc_mag_matrix(model: TB_model, gamma: int = 2):
    """
    Create a calculator for the full orbital magnetization matrix.
    
    This calculator computes the orbital magnetization matrix elements M_orb[n,m](k, gamma)
    between bands n and m at k-point k in direction gamma. This is a generalization of
    make_calc_mag_modern() which only computes the diagonal elements (n=m).
    
    The orbital magnetization matrix is based on Eq. (6) in Pezo et al.,
    PHYSICAL REVIEW B 106, 104414 (2022):
    
    M_orb[n,m](k, gamma) = (1/4) * Im * Σ_{q ≠ n, m} {
        (1/(E_qk - E_nk)) * (v_α[n,q] * v_β[q,m] - v_β[n,q] * v_α[q,m]) +
        (1/(E_qk - E_mk)) * (v_α[n,q] * v_β[q,m] - v_β[n,q] * v_α[q,m])
    }
    
    where α and β are determined by the Levi-Civita symbol for the given γ direction.
    For γ=0 (x): (α,β)=(1,2), for γ=1 (y): (α,β)=(2,0), for γ=2 (z): (α,β)=(0,1).
    
    Args:
        model (TB_model): The tight-binding model class instance containing model parameters.
        gamma (int, optional): Direction of magnetization (0=x, 1=y, 2=z). Default is 2 (z).
    
    Returns:
        function: A JIT-compiled calculator function with signature:
                  (E_k, Vec_k, velocity_k) -> magnetization_matrix
                  where:
                    - E_k (jnp.ndarray): Eigenvalues, shape (nwan,)
                    - Vec_k (jnp.ndarray): Eigenvectors, shape (nwan, nwan)
                    - velocity_k (jnp.ndarray): Velocity operator, shape (nwan, nwan, 3)
                  Returns:
                    magnetization_matrix (jnp.ndarray): M[n,m] matrix, shape (nwan, nwan)
    
    Notes:
        - The diagonal elements (n=m) give the same result as make_calc_mag_modern().
        - Off-diagonal elements (n≠m) represent interband orbital magnetization.
        - A small imaginary part (1e-12) is added to energy denominators to avoid division by zero.
        - The result is purely imaginary, so we take the imaginary part to get the physical quantity.
    
    References:
        Pezo et al., PHYSICAL REVIEW B 106, 104414 (2022), Eq. (6)
    
    Example:
        >>> # Calculate full orbital magnetization matrix in z-direction
        >>> model = read_TB_model("TB.npz")
        >>> calc_mag_matrix = make_calc_mag_matrix(model, gamma=2)
        >>> mag_matrix = calc_mag_matrix(E_k, Vec_k, velocity_k)
    """
    nwan = model.nwan
    alpha, beta = levi_civita[gamma]

    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        """
        Calculate the orbital magnetization matrix for a given k-point.
        
        Args:
            E_k (jnp.ndarray): Eigenvalues of H(k), shape (nwan,).
            Vec_k (jnp.ndarray): Eigenvectors of H(k), shape (nwan, nwan). Columns are eigenvectors.
            velocity_k (jnp.ndarray): Velocity operator in orbital basis, shape (nwan, nwan, 3).
        
        Returns:
            jnp.ndarray: The orbital magnetization matrix M[n,m], shape (nwan, nwan).
        """
        Vec_k_dagger = Vec_k.conj().T
        
        # Transform velocity operators to band basis: v_band = U^† * v_orb * U
        v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        v_beta = Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k
        
        # Remove diagonal elements (no intraband contribution)
        v_alpha_0 = jnp.fill_diagonal(v_alpha, 0.0, inplace=False)
        v_beta_0 = jnp.fill_diagonal(v_beta, 0.0, inplace=False)
        
        # Energy differences: Ediff[q, n] = E_q - E_n
        Ediff = E_k[:,None] - E_k[None,:]
        
        # Set diagonal elements to infinity to avoid division by zero
        # This effectively excludes q=n terms in the summation
        Ediff = jnp.fill_diagonal(Ediff, jnp.inf, inplace=False)
        
        # Green's function: G[q, n] = 1/(E_q - E_n)
        # Small imaginary part (1e-12j) ensures numerical stability
        G = 1.0 / (Ediff + 1e-12 * 1j)
        
        
        # Part 1: Terms involving 1/(E_q - E_n)
        # Σ_q G[q,n] * (v_α[n,q] * v_β[q,m] - v_β[n,q] * v_α[q,m])
        term_G_qn = (
            jnp.einsum("qn,nq,qm->nm", G, v_alpha_0, v_beta_0) 
            - jnp.einsum("qn,nq,qm->nm", G, v_beta_0, v_alpha_0) 
        )
        
        # Part 2: Terms involving 1/(E_q - E_m)
        # Σ_q G[q,m] * (v_α[n,q] * v_β[q,m] - v_β[n,q] * v_α[q,m])
        # Note: This excludes q=m and q=n terms
        term_G_qm = (
            jnp.einsum("qm,nq,qm->nm", G, v_alpha_0, v_beta_0) 
            - jnp.einsum("qm,nq,qm->nm", G, v_beta_0, v_alpha_0) 
        )
        
        # Combine terms: M[n,m] = (1/4) * Im(Part1 + Part2)
        magnetism_matrix = (1/4) * jnp.imag(term_G_qn + term_G_qm)
        return magnetism_matrix
    return _calculator

def make_calc_OBC_modern(
    model: TB_model,
    alpha: int = 0,  # 1st direction of Kubo formula
    beta: int = 1,   # 2nd direction of Kubo formula
    gamma: int = 2,  # direction of orbital magnetization moment
    Gamma: float = 0.025,  # smearing width, in eV unit
    ):
    """
    Create a calculator for the Orbital Berry Curvature (OBC) using modern theory.
    
    This calculator computes the orbital Berry curvature for orbital Hall conductivity (OHC)
    calculations, based on the modern theory of orbital magnetization by Pezo et al.
    [PHYSICAL REVIEW B 106, 104414 (2022)], Eqs. (8) and (9).
    
    The orbital Berry curvature O_BC[n,k,α,β,γ] is defined as:
    
    O_BC[n,k,α,β,γ] = -2 * Im Σ_{m≠n} [1/(E_mk - E_nk)^2] * 
                       [Σ_q (v_α[n,q] * M_γ[q,m] + M_γ[n,q] * v_α[q,m])/2 * v_β[m,n] ]
                    = -2 * Im Σ_{m≠n} [1/(E_mk - E_nk)^2] * 0.5{v_α, M_γ}[n,m] * v_β[m,n]
    where:
    - α, β: directions in the Kubo formula (response and perturbation directions)
    - γ: direction of the orbital magnetization moment
    - M_γ: orbital magnetization matrix in direction γ
    - v_α, v_β: velocity operators in directions α and β
    - Γ (Gamma): Lorentzian broadening parameter for energy denominators
    
    Note: A negative sign is included in front of the OBC, which differs from the
    original formula in Pezo et al. 
    
    Args:
        model (TB_model): The tight-binding model class instance containing model parameters.
        alpha (int, optional): First direction index (0=x, 1=y, 2=z). Default is 0.
        beta (int, optional): Second direction index (0=x, 1=y, 2=z). Default is 1.
        gamma (int, optional): Magnetization direction (0=x, 1=y, 2=z). Default is 2 (z).
        Gamma (float, optional): Smearing width (broadening) in eV. Default is 0.025 eV.
    
    Returns:
        function: A JIT-compiled calculator function with signature:
                  (E_k, Vec_k, velocity_k) -> orbital_berry_curvature
                  where:
                    - E_k (jnp.ndarray): Eigenvalues, shape (nwan,)
                    - Vec_k (jnp.ndarray): Eigenvectors, shape (nwan, nwan)
                    - velocity_k (jnp.ndarray): Velocity operator, shape (nwan, nwan, 3)
                  Returns:
                    orbital_berry_curvature (jnp.ndarray): O_BC[n] for each band, shape (nwan,)
    
    Notes:
        - This calculator internally uses make_calc_mag_matrix() to compute M_γ.
        - The smearing parameter Γ controls the energy window for interband transitions.
        - The result is purely imaginary, so we take the imaginary part to get the physical quantity.
        - For orbital Hall conductivity, integrate over k-space and sum over occupied bands.
    
    References:
        Pezo et al., PHYSICAL REVIEW B 106, 104414 (2022)
    
    Example:
        >>> # Calculate orbital Berry curvature for OHC (α=0, β=1, γ=2)
        >>> model = read_TB_model("TB.npz")
        >>> calc_obc = make_calc_OBC_modern(model, alpha=0, beta=1, gamma=2, Gamma=0.025)
        >>> obc = calc_obc(E_k, Vec_k, velocity_k)
    """
    nwan = model.nwan
    # Create calculator for orbital magnetization matrix
    mag_calculator = make_calc_mag_matrix(model, gamma)

    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        """
        Calculate the Orbital Berry Curvature for a given k-point.
        
        Args:
            E_k (jnp.ndarray): Eigenvalues of H(k), shape (nwan,).
            Vec_k (jnp.ndarray): Eigenvectors of H(k), shape (nwan, nwan). Columns are eigenvectors.
            velocity_k (jnp.ndarray): Velocity operator in orbital basis, shape (nwan, nwan, 3).
        
        Returns:
            jnp.ndarray: The orbital Berry curvature for each band, shape (nwan,).
        """
        Vec_k_dagger = Vec_k.conj().T
        
        # Transform velocity operators to band basis
        v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        v_beta = Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k
        
        # Calculate orbital magnetization matrix (Eq. 6 in Pezo et al.)
        mag_matrix = mag_calculator(E_k, Vec_k, velocity_k)
        
        # Compute symmetrized current operator J_α = (v_α @ M_γ + M_γ @ v_α)/2
        # This is Eq. (9) in Pezo et al.
        J_alpha = (1/2) * (v_alpha @ mag_matrix + mag_matrix @ v_alpha)
        
        # Energy denominator with Lorentzian broadening: 1/[(E_m - E_n)^2 + Γ^2]
        Ediff2 = (E_k[:,None] - E_k[None,:])**2 + Gamma**2 + 1e-12
        # Set diagonal elements to 0 to exclude m=n terms (no intraband contribution)
        G = jnp.fill_diagonal(1.0/Ediff2, 0.0, inplace=False)
        
        # Compute orbital Berry curvature: O_BC[n] = -2 * Im[Σ_m G[m,n] * J_α[n,m] * v_β[m,n]]
        # Using einsum: "nm,nm,mn->n" means Σ_{n,m} G[n,m] * J_α[n,m] * v_β[m,n]
        OBC = -2 * jnp.imag(jnp.einsum("nm,nm,mn->n", G, J_alpha, v_beta))
        
        return OBC
    return _calculator


# a simpler version of kubo calculator
def make_calc_kubo(
    model: TB_model,
    alpha: int = 0,  # 1st direction of kubo formula
    beta: int = 1,   # 2nd direction of kubo formula
    Gamma: float = 0.025,  # smearing width, in eV unit
    Operator: jnp.ndarray = None,  # default is None, then calc Berry Curvature
    ):
    """
    Create a calculator for the Kubo linear response function.
    
    This function implements the Kubo formula for calculating linear response coefficients
    in tight-binding models. It can compute various transport coefficients including:
    
    1. **Berry Curvature** (intrinsic anomalous Hall conductivity) when Operator=None
    2. **Operator-specific Berry Curvature** (e.g., spin Hall conductivity) when Operator is provided
    
    The Kubo formula calculates the response function as:
    
    Ω_n(α,β) = -2 * Im[ Σ_{m≠n} <n|J_α|m><m|v_β|n> / ((E_n - E_m)^2 + Γ^2) ]
    
    where:
    - J_α is the current operator (symmetrized if Operator is provided)
    - v_β is the velocity operator in the β direction
    - Γ is the Lorentzian broadening parameter
    - The sum runs over all bands m ≠ n
    
    **Physical Interpretation:**
    
    - For α=β=x,y,z and Operator=None: Computes the Berry curvature tensor component Ω_xy
    - For Operator=σ_z (spin operator): Computes the spin Berry curvature for spin Hall effect
    - The result is related to the Hall conductivity via σ_xy = (e²/ℏ) * Σ_n f_n * Ω_n
    
    **Mathematical Details:**
    
    The calculation follows these steps:
    
    1. Transform velocity operators to band basis:
       v_α^band = U^† v_α^orb U
       v_β^band = U^† v_β^orb U
    
    2. If Operator is provided, construct symmetrized current:
       J_O = 1/2 (Operator · v_α + v_α · Operator)
       Then transform to band basis: J_O^band = U^† J_O U
    
    3. Compute energy denominator with Lorentzian broadening:
       G_{nm} = 1 / [(E_n - E_m)^2 + Γ^2]
       (diagonal elements set to zero to exclude m=n terms)
    
    4. Calculate the Kubo sum:
       Ω_n = Σ_{m≠n} G_{nm} * <n|J_α|m> * <m|v_β|n>
    
    5. Return -2 * Im(Ω_n) to get the physical quantity
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance containing system parameters.
    
    alpha : int, optional
        First direction index in the Kubo formula (response direction).
        - 0: x-direction
        - 1: y-direction
        - 2: z-direction
        Default is 0 (x-direction).
    
    beta : int, optional
        Second direction index in the Kubo formula (perturbation direction).
        - 0: x-direction
        - 1: y-direction
        - 2: z-direction
        Default is 1 (y-direction).
    
    Gamma : float, optional
        Lorentzian broadening parameter (smearing width) in eV.
        This parameter controls the energy denominator regularization:
        - Larger Gamma: smoother but less accurate results
        - Smaller Gamma: sharper but potentially numerically unstable
        Typical values range from 0.01 to 0.1 eV.
        Default is 0.025 eV.
    
    Operator : jnp.ndarray, optional
        Optional operator matrix in the orbital basis, shape (nwan, nwan).
        If provided, calculates the Berry curvature associated with this operator
        using the symmetrized current operator J_O = 1/2{Operator, v_α}.
        
        Default is None.
    
    **Returns:**
    
    function
        A jittable calculator function `_calculator(E_k, Vec_k, velocity_k)` that
        computes the Kubo formula term for a single k-point.
        
        The returned function has signature:
        ```
        _calculator(E_k, Vec_k, velocity_k) -> jnp.ndarray
        ```
        
        Parameters:
            E_k : jnp.ndarray, shape (nwan,)
                Eigenvalues of the Hamiltonian at k-point.
            Vec_k : jnp.ndarray, shape (nwan, nwan)
                Eigenvectors of the Hamiltonian at k-point (columns are eigenvectors).
            velocity_k : jnp.ndarray, shape (nwan, nwan, 3)
                Velocity operator in orbital basis for all three directions.
        
        Returns:
            jnp.ndarray, shape (nwan,)
                The calculated Kubo term (e.g., Berry curvature) for each band.
    
    **Notes:**
    
    1. **Energy Denominator Regularization:**
       The Lorentzian broadening Γ is essential for numerical stability, especially
       near band crossings or degeneracies. The choice of Γ affects the accuracy
       of the calculation and should be chosen carefully based on the system.
    
    2. **Symmetrized Current Operator:**
       When Operator is provided, the current operator is symmetrized as
       J_O = 1/2{Operator, v_α} to ensure Hermiticity and physical consistency.
       This is particularly important for spin and orbital operators.
    
    3. **Band Summation:**
       The calculation sums over all bands m ≠ n. The diagonal terms (m=n) are
       excluded as they don't contribute to interband transitions in the linear
       response regime.
      
    """
    nwan = model.nwan
    if Operator is None:
        print("calculate Berry Curvature")
    else:
        assert Operator.shape == (nwan, nwan), f"Operator shape {Operator.shape} is not (nwan,nwan)"

    def _calculator(
        E_k: jnp.ndarray,
        Vec_k: jnp.ndarray,
        velocity_k: jnp.ndarray,
        ):
        """
        Calculate the Kubo formula term for a given k-point.
        
        This function computes the linear response coefficient using the Kubo formula.
        The calculation involves transforming operators to the band basis and
        summing over interband transitions with Lorentzian broadening.
        
        Args:
            E_k: jnp.ndarray, shape (nwan,)
                Eigenvalues of the Hamiltonian H(k) at the given k-point.
                These represent the band energies E_n(k).
            
            Vec_k: jnp.ndarray, shape (nwan, nwan)
                Eigenvectors of the Hamiltonian H(k) at the given k-point.
                Each column corresponds to an eigenvector |n(k)>.
                The eigenvectors satisfy H(k)|n(k)> = E_n(k)|n(k)>.
            
            velocity_k: jnp.ndarray, shape (nwan, nwan, 3)
                Velocity operator in the orbital basis for all three directions.
                velocity_k[:,:,α] gives the velocity operator v_α in the α direction.
                The velocity operator is defined as v_α = ∂H(k)/∂k_α.
        
        Returns:
            jnp.ndarray, shape (nwan,)
                The calculated Kubo term (e.g., Berry curvature) for each band.
                The result is -2 * Im[Ω_n] where Ω_n is the Kubo sum.
                For Berry curvature, this gives Ω_n(α,β) for each band n.
        
        Notes:
            - The calculation uses Lorentzian broadening with parameter Gamma
            - Diagonal elements (m=n) are excluded from the sum
            - The result is purely imaginary, and we take -2 * Im(...) to get
              a real physical quantity
        """
        Vec_k_dagger = Vec_k.conj().T
        
        # Transform the beta-component of velocity to the band basis
        # v_beta_band = U^† * v_beta_orb * U
        # where U is the eigenvector matrix and v_beta_orb is the velocity
        # operator in the orbital basis
        v_beta = Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k
        # v_alpha will be calculated later depending on whether Operator is present

        # Calculate the energy denominator factor with Lorentzian broadening
        # denominator[n, m] = (E_n - E_m)^2 + Gamma^2
        # This regularizes the energy denominator to avoid singularities
        # at band crossings or degeneracies
        denominator = (E_k[:,None] - E_k[None,:])**2 + Gamma**2 + 1e-12
        
        # G[n, m] = 1 / ((E_n - E_m)^2 + Gamma^2)
        # This is the Lorentzian broadening factor for interband transitions
        # We exclude diagonal elements (n=m) by setting them to 0.0, as they
        # don't contribute to interband transitions in the linear response regime
        G = jnp.fill_diagonal(1.0/denominator, 0.0, inplace=False)

        if Operator is None:
            # Pure Berry Curvature calculation
            # v_alpha_band = U^† * v_alpha_orb * U
            # Transform the alpha-component of velocity to band basis
            v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        else:
            # Berry Curvature of an Operator (e.g., Spin Hall)
            # Construct the symmetrized current operator: J_O = 1/2 * {Operator, v_alpha}
            # The symmetrization ensures Hermiticity and physical consistency
            J_O = 0.5 * (Operator @ velocity_k[:,:,alpha] + velocity_k[:,:,alpha] @ Operator)
            # Transform J_O to the band basis: J_O_band = U^† * J_O * U
            v_alpha = Vec_k_dagger @ J_O @ Vec_k

        # Calculate the Kubo formula sum:
        # Omega_n = Sum_{m != n} G[n,m] * <n|v_alpha|m> * <m|v_beta|n>
        # 
        # Physical interpretation:
        # - G[n,m]: Lorentzian broadening factor for transition n -> m
        # - <n|v_alpha|m>: Matrix element of current operator between bands n and m
        # - <m|v_beta|n>: Matrix element of velocity operator between bands m and n
        # - The sum runs over all bands m != n (interband transitions only)
        # 
        # Method 1: Using einsum (Compact and explicit)
        # "nm,nm,mn->n":
        #   G[n,m]      -> 1/((E_n-E_m)^2 + Gamma^2)  (Lorentzian factor)
        #   v_alpha[n,m] -> <n|v_alpha|m>             (current matrix element)
        #   v_beta[m,n]  -> <m|v_beta|n>              (velocity matrix element)
        #   ->n          -> Sum over m for each n     (sum over final states)
        Omega_array = jnp.einsum("nm,nm,mn->n", G, v_alpha, v_beta) 
        
        # Method 2: Using element-wise multiplication and sum (Alternative, mathematically equivalent)
        # part_sum = G * v_alpha * v_beta.T  # v_beta.T[n,m] is v_beta[m,n]
        # Omega_array = jnp.sum(part_sum, axis=1)
        
        # The result is -2 * Im(Omega_array) to get the physical quantity
        # The factor of 2 comes from the definition of the Berry curvature
        # and the imaginary part ensures we get a real physical quantity
        return -2 * jnp.imag(Omega_array)
    return _calculator

def make_calc_intra(model:TB_model,
                beta:int=2, # direction of E field
                Operator:jnp.ndarray=None,
                ):
    """
    Create a calculator for intraband contribution to an operator.
    
    This function implements the intraband contribution formula from modern theory
    of orbital magnetism (eq. 3 in arXiv:2509.03026). The intraband contribution
    for band n in direction beta is:
    
    ```
    O_intra(k,n,beta) = <u(n,k)|v_beta|u(n,k)> * <u(n,k)|O|u(n,k)>
    ```
    
    **Important:** This calculator must be used with scanner_type='FD_deri' in
    make_calc_all to compute Fermi surface integrals properly.
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance.
    
    beta : int, optional
        Direction of the electric field (0=x, 1=y, 2=z).
        Default is 2 (z-direction).
    
    Operator : jnp.ndarray, optional
        Operator matrix in orbital basis, shape (nwan, nwan).
        If None, uses identity matrix.
        Default is None.
    
    **Returns:**
    
    function
        A calculator function with signature:
        ```
        _calculator(E_k, Vec_k, Vel_k) -> jnp.ndarray
        ```
        Returns intraband contribution for each band, shape (nwan,).
    """

    print("calculate intraband contribution, be sure use 'FD_deri' scanner")

    nwan = model.nwan
    assert beta in [0,1,2], "beta must be 0, 1, or 2"
    if Operator is None:
        print("Warning: Operator is None, use identity matrix instead")
        Operator = jnp.eye(nwan)
    else:
        assert Operator.shape == (nwan,nwan), f"Operator shape must be ({nwan},{nwan}), but got {Operator.shape}"
    
    def _calculator(E_k:jnp.ndarray, Vec_k:jnp.ndarray, Vel_k:jnp.ndarray):
        """
        Calculate intraband contribution for each band.
        
        Args:
            E_k: Eigenvalues, shape (nwan,). Unused, kept for interface.
            Vec_k: Eigenvectors, shape (nwan, nwan).
            Vel_k: Velocity operator, shape (nwan, nwan, 3).
        
        Returns:
            jnp.ndarray: Intraband contribution for each band, shape (nwan,).
        """
        Vec_k_dagger = Vec_k.conj().T
        v_beta = jnp.einsum('ij,jk,ki->i', Vec_k_dagger, Vel_k[:,:,beta], Vec_k) # nwan
        O_term = jnp.einsum('ij,jk,ki->i', Vec_k_dagger, Operator, Vec_k) # nwan

        intra = v_beta * O_term

        return intra
    return _calculator
        



# K-dependent calculators
# ======================================================
# jax.jit only use for k-dependent calculators

def make_calc_all(model:TB_model,
                kbT:float=0.026, # room temperature
                use_A:bool=False,
                scanner_type:str="FD", # Fermi-Dirac scanner type: "FD" or "FD_deri"
                ):
    """
    Create a unified calculator to compute all registered calculators for k-points.
    
    This function generates a single calculator function that can compute energy,
    velocity, and all user-registered calculators (e.g., Berry curvature, orbital
    magnetization) for a given k-point. It supports Fermi level scanning to compute
    temperature-dependent properties.
    
    **Features:**
    
    - Automatically computes Hamiltonian and velocity operators
    - Runs all calculators registered in model.all_calculators
    - Optional Fermi level scanning with Fermi-Dirac distribution
    - JIT-compiled for efficient computation over k-point meshes
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance. Must have:
        - model.nwan: number of Wannier orbitals
        - model.all_calculators: dict of registered calculators
        - model.E_fermis: list of Fermi levels (required if kbT > 0)
    
    kbT : float, optional
        Temperature in eV (kB * T). Controls Fermi-Dirac broadening.
        - If kbT > 1e-9: performs Fermi level scanning
        - If kbT <= 1e-9: returns results for all bands
        Default is 0.026 eV (room temperature, ~300 K).
    
    use_A : bool, optional
        Whether to include vector potential A in velocity calculation.
        Default is False.
    
    scanner_type : str, optional
        Type of Fermi-Dirac scanner to use for weighting.
        - "FD": use standard Fermi-Dirac distribution f(E) = 1/(exp((E-mu)/kBT) + 1)
        - "FD_deri": use Fermi-Dirac derivative -f*(1-f)/kBT for Fermi surface integration
        Default is "FD".
    
    **Returns:**
    
    function
        A JIT-compiled calculator function with signature:
        ```
        _calculator(k) -> jnp.ndarray
        ```
        
        Parameters:
            k : jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            jnp.ndarray
                If kbT <= 1e-9: shape (1 + n_calculators, nwan)
                    - Row 0: band energies E_n(k)
                    - Row i+1: results of calculator i for each band
                
                If kbT > 1e-9: shape (1 + n_calculators, n_fermi)
                    - Row 0: total energy weighted by Fermi-Dirac
                    - Row i+1: calculator results summed over bands
                    - Columns correspond to different Fermi levels
    
    **Notes:**
    
    1. **Calculator Registration:**
       Calculators must be registered in model.all_calculators before calling
       this function. Each calculator should have signature:
       ```
       calculator(E_k, Vec_k, velocity_k) -> jnp.ndarray, shape (nwan,)
       ```
    
    2. **Fermi Level Scanning:**
       When kbT > 1e-9, results are computed as:
       ```
       result(E_F) = Σ_n f_n(E_F, kbT) * calculator_result_n
       ```
       where f_n is determined by the scanner_type parameter:
       - scanner_type="FD": standard Fermi-Dirac distribution
       - scanner_type="FD_deri": Fermi-Dirac derivative (for DOS calculations)
    
    3. **Performance:**
       The returned function is JIT-compiled with JAX for optimal performance.
       First call will trigger compilation; subsequent calls are fast.
    
    4. **Output Structure:**
       - First row always contains energy information
       - Subsequent rows contain results in the order they appear in
         model.all_calculators.keys()
    
    
    """
    nwan = model.nwan
    all_calculators_name = list(model.all_calculators.keys())
    all_calculators_func = list(model.all_calculators.values())

    shape1 = len(all_calculators_name) + 1
    if len(all_calculators_name) == 0:
        print("no calculator found, calculate energy only")
        calculator_Hk = make_k_hamiltonian_calculator(model)
    else:
        print(f"calculate the following {shape1-1} calculators: {all_calculators_name}")
        calculator_Hk = make_k_hamiltonian_velocity_calculator(model, use_A)
    
    if kbT > 1e-9:
        scan = True
        print(f"calculate the fermi scan with kbT = {kbT}")
        E_fermis = model.E_fermis
        assert model.E_fermis is not None
        
        scanner_type = scanner_type.lower()
        if scanner_type == "fd":
            scanner_func = fermi_dirac_vamp
            print("using standard Fermi-Dirac distribution scanner")
        elif scanner_type == "fd_deri":
            scanner_func = fermi_dirac_derivative_vamp
            print("using Fermi-Dirac derivative scanner")
        else:
            raise ValueError(f"Invalid scanner_type: {scanner_type}. Must be 'FD' or 'FD_deri'")
    else: 
        scan = False
        print("no fermi scan, output all bands")


    # jit here
    @jax.jit
    def _calculator(k: jnp.ndarray):
        """
        Calculate all registered calculators for a single k-point.
        
        This function computes the Hamiltonian, velocity operators, and all
        user-registered calculators for the given k-point. If Fermi scanning
        is enabled, results are weighted by the Fermi-Dirac distribution.
        
        Args:
            k: jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            jnp.ndarray
                If scan=False: shape (1 + n_calculators, nwan)
                    - Row 0: band energies E_n(k)
                    - Row i+1: results of calculator i for each band n
                
                If scan=True: shape (1 + n_calculators, n_fermi)
                    - Row 0: total energy Σ_n f_n(E_F) * E_n(k)
                    - Row i+1: Σ_n f_n(E_F) * calculator_result_i_n
                    - Columns correspond to different Fermi levels in E_fermis
        """
        # n_calc+1 * nwan
        result_bands = jnp.zeros((shape1, nwan), dtype=DEFAULT_TYPE_REAL)
        if shape1 == 1:
            # Only calculate energy (no calculators registered)
            Hk = calculator_Hk(k)
            E_k = jnp.linalg.eigvalsh(Hk)  # nwan
            result_bands = result_bands.at[0].set(E_k)
        else:
            # Calculate Hamiltonian and velocity
            Hk, velocity_k = calculator_Hk(k)
            # Diagonalize to get eigenvalues and eigenvectors
            E_k, Vec_k = jnp.linalg.eigh(Hk)
            # Store band energies
            result_bands = result_bands.at[0].set(E_k)
            # Run all registered calculators
            for i, func in enumerate(all_calculators_func):
                result_bands = result_bands.at[i+1].set(func(E_k, Vec_k, velocity_k))
        
        if not scan:
            # Return results for all bands
            return result_bands
        else:
            # Apply Fermi-Dirac weighting using selected scanner
            f_distribution = scanner_func(E_k, E_fermis, kbT)  # nwan * len(E_fermis)
            # Weight each calculator result by Fermi-Dirac distribution
            result_scan = result_bands[:, :, None] * f_distribution[None, :, :]  # n_calc+1 * nwan * len(E_fermi)
            # Sum over bands to get Fermi-level-dependent results
            return jnp.sum(result_scan, axis=1)  # n_calc+1 * len(E_fermi), sum over nwan
    return _calculator


# scanning Morb
def make_calc_mag_scan(model:TB_model,
                kbT:float=0.026, # room temperature
                use_A:bool=False,
                gamma:int=2, # the direction of the magnetism, default is 2(z direction)
                ):
    '''
    make a function to run a calculator for a list of k-points
    param model: TB_model, # the model class
    return: _calculator: function, # a function to run Mag scanning for a k-point
    '''        
    nwan = model.nwan
    #alpha,beta = levi_civita[gamma]

    #BC_calculator = make_calc_kubo(model, alpha=alpha, beta=beta,Gamma=0.0)
    #Mag_calculator = make_calc_mag_modern(model, gamma=gamma,use_Pezo=False)

    calculator_Hk = make_k_hamiltonian_velocity_calculator(model,use_A)
    E_fermis = model.E_fermis
    print(f"scaning magnetism with  Emin = {E_fermis.min()},Emax = {E_fermis.max()},kbT = {kbT}")

    # split the magnetism calculation into two parts
    # internal function to calculate the magnetism
    def _split_calculator(E_k, Vec_k, velocity_k):
        # ... standard setup ...
        Ediff = E_k[:,None] - E_k[None,:]
        denominator = Ediff**2 + 1e-12 
        Vec_k_dagger = Vec_k.conj().T
        
        # --- Part A: The Energy Dependent Term ---
        # Numerator = E_q + E_n
        num_A = E_k[None,:] + E_k[:,None] 
        Gqn_A = num_A / denominator
        Gqn_A = jnp.fill_diagonal(Gqn_A, 0.0, inplace=False)
        
        # --- Part B: The Constant Term (Berry-like) ---
        # Numerator = 1
        Gqn_B = 1.0 / denominator
        Gqn_B = jnp.fill_diagonal(Gqn_B, 0.0, inplace=False)

        # Calculate Cross Products (Same as before)
        v = jnp.einsum("ij,jkm,kl->mil", Vec_k_dagger, velocity_k, Vec_k)

        mag_A = jnp.zeros((3,nwan))
        mag_B = jnp.zeros((3,nwan))
        # We need Im[ sum G * v_a * v_b ]
        # Compute the cross product part common to both
        for gamma in  range(3):
            alpha,beta = levi_civita[gamma]
            cross_A_1 = jnp.einsum("qn,nq,qn->n", Gqn_A, v[alpha], v[beta], optimize='optimal')
            cross_A_2 = jnp.einsum("qn,nq,qn->n", Gqn_A, v[beta], v[alpha], optimize='optimal')
            mag_A_gamma = jnp.imag((cross_A_1 - cross_A_2) / 2)
            mag_A = mag_A.at[gamma].set(mag_A_gamma)

            cross_B_1 = jnp.einsum("qn,nq,qn->n", Gqn_B, v[alpha], v[beta], optimize='optimal')
            cross_B_2 = jnp.einsum("qn,nq,qn->n", Gqn_B, v[beta], v[alpha], optimize='optimal')
            mag_B_gamma = jnp.imag((cross_B_1 - cross_B_2) / 2)
            mag_B = mag_B.at[gamma].set(mag_B_gamma)
        return mag_A, mag_B # 3 * nwan

    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate all calculators for a k-point
        para:
            k: jnp.ndarray, # 3, k-point in reciprocal space
        return:
            results: jnp.ndarray, # shape in (n_fermi), 
            results of Orbit Mag scanning 
        '''
        Hk, velocity_k = calculator_Hk(k)
        E_k,Vec_k = jnp.linalg.eigh(Hk)


        # 2 ways to calculate the magnetism

        mag_A, mag_B = _split_calculator(E_k,Vec_k,velocity_k) # 3 * nwan
        # alternative way to calculate the magnetism: 
        # using existing BC and Mag calculator
        #BC = BC_calculator(E_k,Vec_k,velocity_k)
        #mag = Mag_calculator(E_k,Vec_k,velocity_k)
        
        f_distribution = fermi_dirac_vamp(E_k,E_fermis,kbT) # nwan * len(E_fermis)

        Term_A = f_distribution[None,:,:] * mag_A[:,:,None] # 3 *  nwan * len(E_fermis)  
        Term_B = f_distribution[None,:,:] * mag_B[:,:,None] * E_fermis[None,None,:]# 3 * nwan * len(E_fermis) 
        result_scan = Term_A - Term_B * 2 # 3 *  nwan * len(E_fermis)
        return jnp.sum(result_scan,axis=1) # 3 *  len(E_fermis),

        # The way that use BC_calculator and Mag_calculator
        # Term_A = f_distribution * mag[:,None]
        # Term_B = f_distribution * BC[:,None] * E_fermis[None,:]
        # result_scan = Term_A + Term_B  # len(E_fermis),
        # return jnp.sum(result_scan,axis=0) #  len(E_fermis),
    return _calculator
# Calculators==================End



# testing version
# Only for testing, not for production use
def make_calc_mag_test(model: TB_model):
    '''
    make a calculator for the magnetism of the model
    using testing version 
    '''
    AA_calc = make_calc_AA_k(model)
    calculator_Hk = make_k_hamiltonian_velocity_calculator(model,use_A=False)
    nwan = model.nwan
    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate the magnetism of the model
        para:
            k: jnp.ndarray, # 3
        return:
            magnetism: jnp.ndarray, # nwan
        '''
        Hk, velocity_k = calculator_Hk(k)
        E_k, Vec_k = jnp.linalg.eigh(Hk)
        Vec_k_dagger = Vec_k.conj().T
        AA_k_W = AA_calc(k)
        #Vec_k_dagger @ AA_k_W @ Vec_k, shape (nwan,nwan,3)
        AA_B = jnp.einsum("ij,jka,kl->ila",Vec_k_dagger,AA_k_W,Vec_k)    

        Ediff = E_k[:,None] - E_k[None,:] # D[q,n] = E_qk - E_nk
        G = Ediff / Ediff**2 + 1e-9  # nwan * nwan
        G = jnp.fill_diagonal(G,0.0,inplace=False)
        
        velocity_H = jnp.einsum("ij,jka,kl->ila",Vec_k_dagger,velocity_k,Vec_k) # nwan * nwan * 3
        D_term = 1j * G[:,:,None] * velocity_H # nwan * nwan * 3

        AA_H = AA_B + D_term # nwan * nwan * 3
        magnetism = jnp.zeros((3,nwan))
        for gamma in  range(3):
            alpha, beta = levi_civita[gamma]
            temp = jnp.einsum("mn,nm,mn->n",Ediff,AA_H[:,:,alpha],AA_H[:,:,beta]).imag
            magnetism = magnetism.at[gamma].set(temp)
            
        return magnetism
    return _calculator


# test the calculator
# test pass
if __name__ == "__main__":
    pass