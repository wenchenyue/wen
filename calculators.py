
import types
import jax
import jax.numpy as jnp
from TB_model import TB_model
from utility import fermi_dirac_vamp, fermi_dirac_derivative_neg_vamp,DEFAULT_TYPE_COMPLEX, DEFAULT_TYPE_REAL
from Hk import make_k_hamiltonian_calculator, make_k_hamiltonian_velocity_calculator


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





def make_calc_inner(model: TB_model, Operator = None, H_gauge: bool = False):
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
        Operator (jnp.ndarray or function): The operator matrix or function. 
                               - If jnp.ndarray: shape (nwan, nwan), in orbital basis
                               - If function: callable with signature (E_k, Vec_k, velocity_k) -> jnp.ndarray
                               - If None: uses identity matrix
        H_gauge (bool): Gauge choice for the operator representation.
                       - If True: Operator is already in the band (Hamiltonian) basis, 
                                  directly take diagonal elements
                       - If False: Operator is in the orbital (Wannier) basis, 
                                   transform to band basis using eigenvectors
                       Default is False.
    
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
        AssertionError: If Operator is jnp.ndarray and shape does not match (nwan, nwan).
        ValueError: If Operator type is not supported.
    
    Notes:
        - The velocity_k parameter is ignored in this calculation but kept for interface consistency.
        - The result is always real because we're computing expectation values of Hermitian operators.
        - After Fermi-Dirac smearing, states below EF contribute 1, above EF contribute 0.
        - When H_gauge=True, Operator is assumed to be diagonal in the band basis (only diagonal elements are taken).
        - When H_gauge=False and Operator is a function, it can be k-dependent.

    """
    nwan = model.nwan

    if Operator is None:
        Operator_func = lambda E_k, Vec_k, Vel_k: jnp.eye(nwan)
    elif isinstance(Operator,jnp.ndarray):
        assert Operator.shape == (nwan,nwan), f"Operator shape should be ({nwan},{nwan}), but got {Operator.shape}"
        Operator_func = lambda E_k, Vec_k, Vel_k: Operator
    elif isinstance(Operator,types.FunctionType): 
        # Only custom functions are supported, built-in functions and callable objects are not.
        Operator_func = Operator
        # useful check for developers to ensure the function signature is correct
        import inspect
        sig = inspect.signature(Operator_func)
        print(f"custom Operator function {Operator_func.__name__}, parameters: {sig}")
    else:
        raise ValueError(f"Operator type {type(Operator)} is not supported. Only None, jnp.ndarray, and custom functions are allowed.")

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
        # where:
        #   Vec_k.conj().T[i,j] = ψ_n*[j] (transpose of conjugate)
        #   Operator[j,k] = O[j,k]
        #   Vec_k[k,i] = ψ_n[i]
        #   Result[i] = Σ_{j,k} ψ_n*[j] * O[j,k] * ψ_n[k] = <ψ_n|O|ψ_n>

        Operator_mat = Operator_func(E_k, Vec_k, velocity_k)
        if H_gauge: # operator in the Hamiltonian(band) basis
            return jnp.real(jnp.diag(Operator_mat))
        else: # operator in the orbital(wannier) basis
            inner_product = jnp.einsum('ij,jk,ki->i', Vec_k.conj().T, Operator_mat, Vec_k, optimize='optimal')
            return jnp.real(inner_product)
    return _calculator



# a simpler version of kubo calculator
def make_calc_kubo(
    model: TB_model,
    alpha: int = 0,  # 1st direction of kubo formula
    beta: int = 1,   # 2nd direction of kubo formula
    Gamma: float = 0.025,  # smearing width, in eV unit
    Operator = None,  # default is None, then calc Berry Curvature
    H_gauge: bool = False,  # default is False, then calc Operator-specific Berry Curvature
    ):
    """
    Create a calculator for the Kubo linear response function.
    
    This function implements the Kubo formula for calculating linear response coefficients
    in tight-binding models. It can compute various transport coefficients including:
    
    1. **Berry Curvature** (intrinsic anomalous Hall conductivity)
    2. **Operator-specific Berry Curvature** (e.g., spin Hall conductivity) when Operator is provided
    
    The Kubo formula calculates the response function as:
    
    Ω_n(α,β) = -2 * Im[ Σ_{m≠n} <n|J_α|m><m|v_β|n> / ((E_n - E_m)^2 + Γ^2) ]
    
    where:
    - J_α is the current operator (symmetrized if Operator is provided)
    - v_β is the velocity operator in the β direction
    - Γ is the Lorentzian broadening parameter
    - The sum runs over all bands m ≠ n
    
    **Physical Interpretation:**
    
    - For α=0, β=1 (x,y) with Operator=I (identity): Computes the Berry curvature tensor component Ω_xy
    - For Operator=σ_z (spin operator): Computes the spin Berry curvature for spin Hall effect
    - The result is related to the Hall conductivity via σ_xy = (e²/ℏ) * Σ_n f_n * Ω_n
    
    **Mathematical Details:**
    
    The calculation follows these steps:
    
    1. Transform velocity operators to band basis:
       v_β^band = U^† v_β^orb U
    
    2. Construct the current operator J_α based on H_gauge parameter:
       - If H_gauge=True (Operator in band basis):
         v_α^band = U^† v_α^orb U
         J_α^band = 1/2 * (Operator_mat @ v_α^band + v_α^band @ Operator_mat)
       - If H_gauge=False (Operator in orbital basis):
         J_O^orb = 1/2 * (Operator_mat @ v_α^orb + v_α^orb @ Operator_mat)
         J_α^band = U^† J_O^orb U
    
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
    
    Operator : jnp.ndarray or function, optional
        Optional operator matrix or function for computing operator-specific Berry curvature.
        - If jnp.ndarray: shape (nwan, nwan), can be in orbital or band basis depending on H_gauge
        - If function: callable with signature (E_k, Vec_k, velocity_k) -> jnp.ndarray
        - If None: uses identity matrix (calculates standard Berry curvature)
        Default is None.
    
    H_gauge : bool, optional
        Gauge choice for the Operator representation.
        - If True: Operator is already in the band (Hamiltonian) basis
        - If False: Operator is in the orbital (Wannier) basis
        Default is False.
    
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
       The current operator is always symmetrized as J_α = 1/2{Operator, v_α} 
       (or 1/2{I, v_α} = v_α when Operator=I) to ensure Hermiticity and 
       physical consistency. The H_gauge parameter determines whether the 
       symmetrization happens in the orbital basis (H_gauge=False) or in the 
       band basis (H_gauge=True).
    
    3. **Band Summation:**
       The calculation sums over all bands m ≠ n. The diagonal terms (m=n) are
       excluded as they don't contribute to interband transitions in the linear
       response regime.
    
    4. **Operator Flexibility:**
       The Operator can be provided as a static matrix (jnp.ndarray) or as a 
       function that returns a k-dependent operator. When using H_gauge=True,
       the Operator is expected to be already in the band basis, while 
       H_gauge=False expects the Operator in the orbital basis.
      
    """
    nwan = model.nwan
    if Operator is None:
        Operator_func = lambda E_k, Vec_k, Vel_k: jnp.eye(nwan)
    elif isinstance(Operator,jnp.ndarray):
        assert Operator.shape == (nwan,nwan), f"Operator shape should be ({nwan},{nwan}), but got {Operator.shape}"
        Operator_func = lambda E_k, Vec_k, Vel_k: Operator
    elif isinstance(Operator,types.FunctionType): 
        # Only custom functions are supported, built-in functions and callable objects are not.
        Operator_func = Operator  
        # useful check for developers to ensure the function signature is correct
        import inspect
        sig = inspect.signature(Operator_func)
        print(f"custom Operator function {Operator_func.__name__}, parameters: {sig}")
    else:
        raise ValueError(f"Operator type {type(Operator)} is not supported. Only None, jnp.ndarray, and custom functions are allowed.")  

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

        Operator_mat = Operator_func(E_k, Vec_k, velocity_k)
        
        # Construct the current operator J_α based on H_gauge parameter
        if H_gauge:
            # Operator is in the Hamiltonian(band) basis
            # First transform v_α to band basis, then symmetrize with Operator
            v_alpha_band = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
            v_alpha = 0.5 * (Operator_mat @ v_alpha_band + v_alpha_band @ Operator_mat)
        else:
            # Operator is in the orbital(wannier) basis
            # First symmetrize Operator with v_α in orbital basis, then transform to band basis
            J_O = 0.5 * (Operator_mat @ velocity_k[:,:,alpha] + velocity_k[:,:,alpha] @ Operator_mat)
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
                beta:int = 2, # direction of E field
                Operator = None,# matrix or function
                H_gauge:bool = False, # operator in the Hamiltonian(band) basis
                ):
    """
    Create a calculator for intraband contribution to an operator.
    
    This function implements the intraband contribution formula from modern theory
    of orbital magnetism (eq. 3 in arXiv:2509.03026). The intraband contribution
    for band n in direction beta is:
    
    ```
    O_intra(k,n,beta) = |<u(n,k)|v_beta|u(n,k)>| * <u(n,k)|O|u(n,k)>
    ```
    
    **Physical Interpretation:**
    This formula represents the intraband (Drude-like) contribution to transport
    properties. The first term ⟨u(n,k)|v_β|u(n,k)⟩ is the intraband velocity,
    and the second term ⟨u(n,k)|O|u(n,k)⟩ is the expectation value of the operator
    O in the band n state.
    
    **Important:** This calculator must be used with scanner_type='FD_deri' in
    make_calc_all to compute Fermi surface integrals properly.
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance.
    
    beta : int, optional
        Direction of the electric field (0=x, 1=y, 2=z).
        Default is 2 (z-direction).
    
    Operator : jnp.ndarray, callable, or None, optional
        Operator matrix or function for computing the operator expectation value.
        - If jnp.ndarray: shape (nwan, nwan), can be in orbital or band basis depending on H_gauge
        - If callable: custom function with signature (E_k, Vec_k, Vel_k) -> jnp.ndarray
        - If None: uses identity matrix (calculates velocity term only)
        Default is None.
    
    H_gauge : bool, optional
        Gauge choice for the Operator representation.
        - If True: Operator is already in the Hamiltonian (band) basis
        - If False: Operator is in the orbital (Wannier) basis and will be transformed
        Default is False.
    
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
    if Operator is None:
        Operator_func = lambda E_k, Vec_k, Vel_k: jnp.eye(nwan)
    elif isinstance(Operator,jnp.ndarray):
        assert Operator.shape == (nwan,nwan), f"Operator shape should be ({nwan},{nwan}), but got {Operator.shape}"
        Operator_func = lambda E_k, Vec_k, Vel_k: Operator
    elif isinstance(Operator,types.FunctionType): 
        # Only custom functions are supported, built-in functions and callable objects are not.
        Operator_func = Operator
        # useful check for developers to ensure the function signature is correct
        import inspect
        sig = inspect.signature(Operator_func)
        print(f"custom Operator function {Operator_func.__name__}, parameters: {sig}")
    else:
        raise ValueError(f"Operator type {type(Operator)} is not supported. Only None, jnp.ndarray, and custom functions are allowed.")

    def _calculator(E_k:jnp.ndarray, Vec_k:jnp.ndarray, Vel_k:jnp.ndarray):
        """
        Calculate intraband contribution for each band.
        
        This function computes the intraband contribution O_intra(n,β) for each band n
        using the formula: O_intra(n,β) = v_β(n) * O(n)
        where v_β(n) = ⟨u(n,k)|v_β|u(n,k)⟩ is the intraband velocity and
        O(n) = ⟨u(n,k)|O|u(n,k)⟩ is the operator expectation value.
        
        Args:
            E_k: jnp.ndarray, shape (nwan,)
                Eigenvalues of the Hamiltonian at k-point.
                Kept for interface consistency but not used in calculation.
            
            Vec_k: jnp.ndarray, shape (nwan, nwan)
                Eigenvectors of the Hamiltonian at k-point.
                Each column corresponds to an eigenvector |u(n,k)⟩.
            
            Vel_k: jnp.ndarray, shape (nwan, nwan, 3)
                Velocity operator in orbital basis for all three directions.
                Vel_k[:,:,β] gives the velocity operator v_β in the β direction.
        
        Returns:
            jnp.ndarray, shape (nwan,)
                Intraband contribution O_intra(n,β) for each band n.
                Computed as the product of intraband velocity and operator expectation value.
        """
        # Get the operator matrix (either static or from custom function)
        Operator_mat = Operator_func(E_k, Vec_k, Vel_k)
        
        # Compute conjugate transpose of eigenvector matrix
        Vec_k_dagger = Vec_k.conj().T
        
        # Calculate intraband velocity v_β(n) = ⟨u(n,k)|v_β|u(n,k)⟩
        # Using einsum: 'ij,jk,ki->i' performs the matrix multiplication
        # and takes diagonal elements to get expectation values
        v_beta = jnp.einsum('ij,jk,ki->i', Vec_k_dagger, Vel_k[:,:,beta], Vec_k)
        # take Absolute value
        v_beta = jnp.abs(v_beta)

        # Calculate operator expectation value O(n) = ⟨u(n,k)|O|u(n,k)⟩
        # based on H_gauge parameter
        if H_gauge:
            # Operator is already in the Hamiltonian (band) basis
            O_term = Operator_mat
        else:
            # Operator is in the orbital (Wannier) basis, need to transform to band basis
            # O(n) = ⟨u(n,k)|O_orb|u(n,k)⟩
            O_term = jnp.einsum('ij,jk,ki->i', Vec_k_dagger, Operator_mat, Vec_k)

        # Calculate intraband contribution: O_intra(n,β) = v_β(n) * O(n)
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
        elif scanner_type == "fd_deri": # negtive derivative of fermi-dirac function
            scanner_func = fermi_dirac_derivative_neg_vamp
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
        
        The calculation proceeds in two main steps:
        1. Diagonalize the Hamiltonian to get band energies and wavefunctions
        2. Apply each registered calculator to obtain physical quantities
        
        If kbT > 1e-9 (Fermi scanning enabled), the results are further weighted
        by the Fermi-Dirac distribution to obtain temperature-dependent properties.
        
        Args:
            k: jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
                Coordinates should be in the same basis as the tight-binding model.
        
        Returns:
            jnp.ndarray
                If scan=False: shape (1 + n_calculators, nwan)
                    - Row 0: band energies E_n(k) for each band n
                    - Row i+1: results of calculator i for each band n
                
                If scan=True: shape (1 + n_calculators, n_fermi)
                    - Row 0: total energy Σ_n f_n(E_F) * E_n(k) weighted by Fermi-Dirac
                    - Row i+1: Σ_n f_n(E_F) * calculator_result_i_n
                    - Columns correspond to different Fermi levels in E_fermis
        """
        # Initialize result array: (n_calculators + 1) × nwan
        # Extra row is for band energies
        result_bands = jnp.zeros((shape1, nwan), dtype=DEFAULT_TYPE_REAL)
        
        if shape1 == 1:
            # Only calculate energy (no calculators registered)
            Hk = calculator_Hk(k)
            E_k = jnp.linalg.eigvalsh(Hk)  # nwan
            result_bands = result_bands.at[0].set(E_k)
        else:
            # Calculate Hamiltonian and velocity operators
            Hk, velocity_k = calculator_Hk(k)
            
            # Diagonalize to get eigenvalues and eigenvectors
            # E_k: band energies E_n(k), Vec_k: eigenvectors |u_n(k)⟩
            E_k, Vec_k = jnp.linalg.eigh(Hk)
            
            # Store band energies in first row
            result_bands = result_bands.at[0].set(E_k)
            
            # Run all registered calculators
            for i, func in enumerate(all_calculators_func):
                # Each calculator computes: calc_n = f(E_k, Vec_k, velocity_k)
                result_bands = result_bands.at[i+1].set(func(E_k, Vec_k, velocity_k))
        
        if not scan:
            # Return results for all bands (no Fermi-Dirac weighting)
            return result_bands
        else:
            # Apply Fermi-Dirac weighting using selected scanner
            # f_distribution[n, E_F] = f_n(E_F, kbT) for band n and Fermi level E_F
            f_distribution = scanner_func(E_k, E_fermis, kbT)  # nwan × len(E_fermis)
            
            # Weight each calculator result by Fermi-Dirac distribution
            # Broadcasting: result_bands[:, :, None] is (n_calc+1) × nwan × 1
            #               f_distribution[None, :, :] is 1 × nwan × len(E_fermi)
            result_scan = result_bands[:, :, None] * f_distribution[None, :, :]
            
            # Sum over bands to get Fermi-level-dependent results
            # Sum along axis=1 (band index) to get total contribution at each E_F
            return jnp.sum(result_scan, axis=1)
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
# Only for testing, not for use
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
            temp = jnp.imag(jnp.einsum("mn,nm,mn->n",Ediff,AA_H[:,:,alpha],AA_H[:,:,beta]))
            magnetism = magnetism.at[gamma].set(temp)
            
        return magnetism
    return _calculator


# test the calculator
# test pass
if __name__ == "__main__":
    pass