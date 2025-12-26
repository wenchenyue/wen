import jax
import jax.numpy as jnp
from TB_model import TB_model



# using double precision
jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64

# Hk Calculators==================Begin

# Hamiltonian in k-space
# Use functional factory pattern to create calculators
# Pass a dataclass because the calculator should be coupled to the model

def make_k_hamiltonian_calculator(model: TB_model):
    """
    Create a calculator for the tight-binding Hamiltonian in k-space.
    
    This function generates a JIT-compiled calculator that computes the Bloch
    Hamiltonian H(k) for a given k-point using the real-space tight-binding
    matrix elements H(R). The calculation accounts for both the unit cell
    positions and the intra-cell positions of Wannier functions.
    
    **Physical Background:**
    
    The Bloch Hamiltonian is constructed from the real-space tight-binding
    matrix elements H_{ij}(R) = <w_i(0)|H|w_j(R)>, where w_i are Wannier
    functions. The k-space Hamiltonian is given by the Fourier transform:
    
    H(k) = Σ_R e^{ik·R} H(R)
    
    where H(R) is the tight-binding matrix for lattice vector R. To account
    for the positions of Wannier functions within the unit cell, we use:
    
    H(k) = Σ_R e^{ik·(R + τ_j - τ_i)} H_{ij}(R)
    
    where τ_i is the position of Wannier function i within the unit cell.
    
    **Mathematical Formulation:**
    
    The calculation proceeds in two steps:
    
    1. Cell phase factor: accounts for lattice vector R
       Phase_R(R) = e^{ik·R}
    
    2. Wannier phase factor: accounts for intra-cell positions
       Phase_W(i,j) = e^{ik·(τ_j - τ_i)}
    
    The final Hamiltonian is:
    H(k) = [Σ_R Phase_R(R) * H(R)] ⊙ Phase_W
    
    where ⊙ denotes element-wise multiplication.
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance containing:
        - model.cell_center: shape (nR, 3), lattice vectors R
        - model.center_matrix: shape (nwan, nwan, 3), τ_j - τ_i for each orbital pair
        - model.HR: shape (nR, nwan, nwan), real-space Hamiltonian matrices
    
    **Returns:**
    
    function
        A JIT-compiled calculator function with signature:
        ```
        _calculator(k) -> Hk
        ```
        
        Parameters:
            k : jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            Hk : jnp.ndarray, shape (nwan, nwan)
                The Bloch Hamiltonian at k-point k.
                The result is Hermitian by construction.
    
    
    """
    cell_center = model.cell_center
    center_matrix = model.center_matrix
    HR = model.HR

    @jax.jit
    def _calculator(k: jnp.ndarray):
        """
        Calculate the Hamiltonian in k-space.
        
        This function computes the Bloch Hamiltonian H(k) by Fourier transforming
        the real-space tight-binding matrix elements, accounting for both
        lattice vectors and intra-cell Wannier function positions.
        
        Args:
            k: jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            Hk: jnp.ndarray, shape (nwan, nwan)
                The Bloch Hamiltonian at k-point k.
                The result is Hermitian: Hk = Hk^†.
        """
        # Calculate the phase factor for each lattice vector R
        # This phase accounts for the periodic boundary conditions
        # Phase_R(R) = e^{ik·R}
        dot_product = jnp.einsum('ij,j->i', cell_center, k)
        Phase_R = jnp.exp(1j * dot_product)  # nR

        # Calculate the phase factor for each Wannier function pair
        # This phase accounts for the different positions of Wannier functions
        # within the unit cell: Phase_W(i,j) = e^{ik·(τ_j - τ_i)}
        dot_product = jnp.einsum('ijk,k->ij', center_matrix, k)
        Phase_W = jnp.exp(1j * dot_product)  # nwan * nwan
        
        # Part 1: Use einsum to calculate the partial sum over lattice vectors
        # partial_sum(i,j) = Σ_R Phase_R(R) * HR(R,i,j)
        # This gives the Fourier transform of H(R) without intra-cell phases
        partial_sum = jnp.einsum('i,ijk->jk', Phase_R, HR)
        
        # Part 2: Multiply by the phase factor for intra-cell positions
        # Hk(i,j) = partial_sum(i,j) * Phase_W(i,j)
        Hk = partial_sum * Phase_W

        # Ensure the Hamiltonian is Hermitian by symmetrization
        # This corrects for any numerical precision issues
        return (Hk + Hk.conj().T) / 2

    return _calculator


# Calculate the anomalous position operator AA_k in Wannier gauge
# This is useful for computing the Berry connection in Wannier basis
def make_calc_AA_k(model: TB_model):
    """
    Create a calculator for the anomalous position operator in k-space (Wannier gauge).
    
    This function generates a calculator for the anomalous position operator A(k),
    which is defined as the Berry connection in the Wannier basis. The anomalous
    position operator is essential for calculating orbital magnetization and other
    Berry-phase-related properties in tight-binding models.
    
    **Physical Background:**
    
    The anomalous position operator A(k) is defined as:
    
    A(k) = -i Σ_R e^{ik·R} [r(R) ⊗ H(R) - H(R) ⊗ r(R)^T]
    
    where r(R) is the position operator matrix element between Wannier functions
    at different unit cells. In the Wannier gauge, this operator captures the
    non-trivial geometry of the Bloch states beyond the standard position operator.
    
    The operator is Hermitian and has three components (x, y, z):
    A(k) = [A_x(k), A_y(k), A_z(k)]
    
    **Mathematical Formulation:**
    
    The calculation follows the same two-step phase factor approach as the
    Hamiltonian:
    
    1. Cell phase factor: Phase_R(R) = e^{ik·R}
    
    2. Wannier phase factor: Phase_W(i,j) = e^{ik·(τ_j - τ_i)}
    
    The anomalous position operator is:
    A(k) = [Σ_R Phase_R(R) * AA(R)] ⊙ Phase_W ⊗ 1_3
    
    where AA(R) is the real-space anomalous position matrix, and the final
    result has shape (nwan, nwan, 3) for the three spatial components.
    
    **Physical Significance:**
    
    - The anomalous position operator appears in the modern theory of orbital
      magnetization (Pezo et al., PRB 2023)
    - It contributes to the velocity operator in the presence of magnetic fields
    - It is related to the Berry connection: A(k) = -i <u(k)|∇_k|u(k)>
    - Essential for calculating orbital magnetic moments
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance containing:
        - model.cell_center: shape (nR, 3), lattice vectors R
        - model.center_matrix: shape (nwan, nwan, 3), τ_j - τ_i for each orbital pair
        - model.AA_R: shape (nR, nwan, nwan, 3), real-space anomalous position matrices
    
    **Returns:**
    
    function
        A JIT-compiled calculator function with signature:
        ```
        _calculator(k) -> AA_k
        ```
        
        Parameters:
            k : jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            AA_k : jnp.ndarray, shape (nwan, nwan, 3)
                The anomalous position operator at k-point k.
                - AA_k[:,:,0]: x-component
                - AA_k[:,:,1]: y-component
                - AA_k[:,:,2]: z-component
                The result is Hermitian for each component.
    
    **Notes:**
    
    1. **Hermiticity:**
       Each component of AA_k is explicitly symmetrized as (AA_k + AA_k^†)/2
       to ensure Hermiticity.
    
    2. **Wannier Gauge:**
       This operator is defined in the Wannier gauge, where the Wannier functions
       are maximally localized. In other gauges, the operator would include
       additional Berry connection terms.
    
    3. **Relation to Berry Connection:**
       The anomalous position operator is related to the Berry connection A(k)
       (not to be confused with the vector potential A in electromagnetism):
       A(k) = -i <u(k)|∇_k|u(k)>

    
    """
    cell_center = model.cell_center
    center_matrix = model.center_matrix
    AA_R = model.AA_R

    @jax.jit
    def _calculator(k: jnp.ndarray):
        """
        Calculate the anomalous position operator in k-space (Wannier gauge).
        
        This function computes the anomalous position operator A(k) by Fourier
        transforming the real-space matrix elements, accounting for both
        lattice vectors and intra-cell Wannier function positions.
        
        Args:
            k: jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            AA_k: jnp.ndarray, shape (nwan, nwan, 3)
                The anomalous position operator at k-point k.
                - AA_k[:,:,0]: x-component
                - AA_k[:,:,1]: y-component
                - AA_k[:,:,2]: z-component
                Each component is Hermitian: AA_k[:,:,α] = AA_k[:,:,α]^†.
        """
        # Calculate the phase factor for each lattice vector R
        # This phase accounts for the periodic boundary conditions
        # Phase_R(R) = e^{ik·R}
        dot_product = jnp.einsum('ij,j->i', cell_center, k)
        Phase_R = jnp.exp(1j * dot_product)  # nR
        
        # Partial sum over lattice vectors for each component
        # partial_sum(i,j,α) = Σ_R Phase_R(R) * AA_R(R,i,j,α)
        partial_sum = jnp.einsum('i,ijkl->jkl', Phase_R, AA_R)
        
        # Calculate the phase factor for each Wannier function pair
        # This phase accounts for the different positions of Wannier functions
        # within the unit cell: Phase_W(i,j) = e^{ik·(τ_j - τ_i)}
        dot_product = jnp.einsum('ijk,k->ij', center_matrix, k)
        Phase_W = jnp.exp(1j * dot_product)  # nwan * nwan
        
        # Multiply by the phase factor for intra-cell positions
        # AA_k(i,j,α) = partial_sum(i,j,α) * Phase_W(i,j)
        # The None indexing broadcasts Phase_W to match the 3rd dimension
        AA_k = partial_sum * Phase_W[:, :, None]
        
        # Ensure AA_k is Hermitian for each spatial component
        # This corrects for any numerical precision issues
        AA_k = (AA_k + AA_k.transpose(1, 0, 2).conj()) / 2
        return AA_k
    return _calculator


# A calculator for the velocity operator of the Hamiltonian with respect to k
# The velocity operator is ∂H(k)/∂k_μ, which describes the group velocity of electrons
# This calculator is coupled to the model, so we pass the model class
def make_k_hamiltonian_velocity_calculator(model: TB_model, use_A: bool = False):
    """
    Create a calculator for the velocity operator of the Hamiltonian in k-space.
    
    This function generates a JIT-compiled calculator that computes the velocity
    operator v_μ(k) = ∂H(k)/∂k_μ, which describes the group velocity of electrons
    in a crystal. The velocity operator is essential for calculating transport
    properties such as conductivity, orbital magnetization, and Berry curvature.
    
    **Physical Background:**
    
    The velocity operator in a crystal is given by the derivative of the Bloch
    Hamiltonian with respect to k:
    
    v_μ(k) = ∂H(k)/∂k_μ
    
    This operator appears in the semiclassical equations of motion for Bloch
    electrons and determines the group velocity of wave packets:
    
    v_n(k) = ∂E_n(k)/∂k = <u_n(k)|v(k)|u_n(k)>
    
    where E_n(k) and |u_n(k)> are the energy and periodic part of the Bloch state
    for band n.
    
    When `use_A=True`, the velocity operator includes the anomalous position
    operator contribution:
    
    v_μ(k) = ∂H(k)/∂k_μ - i[A_μ(k), H(k)]
    
    where A_μ(k) is the anomalous position operator and [·,·] denotes the
    commutator. This form is essential for the modern theory of orbital
    magnetization.
    
    **Mathematical Formulation:**
    
    The velocity operator is computed using numerical differentiation:
    
    ∂H(k)/∂k_μ ≈ [H(k + δe_μ) - H(k - δe_μ)] / (2δ)
    
    where δ is a small finite difference parameter (default: 1e-6) and e_μ is
    the unit vector in the μ direction.
    
    When `use_A=True`, the velocity operator includes the commutator term:
    
    v_μ(k) = ∂H(k)/∂k_μ - i[A_μ(k), H(k)]
           = ∂H(k)/∂k_μ - i[A_μ(k)H(k) - H(k)A_μ(k)]
    
    This ensures the velocity operator is gauge-invariant and correctly
    describes the orbital magnetic moment contribution.
    
    
    **Parameters:**
    
    model : TB_model
        The tight-binding model instance containing all necessary matrices
        for calculating H(k) and optionally A(k).
    
    use_A : bool, optional (default=False)
        Whether to include the anomalous position operator contribution.
        - If False: v_μ(k) = ∂H(k)/∂k_μ
        - If True: v_μ(k) = ∂H(k)/∂k_μ - i[A_μ(k), H(k)]
        
        Set to True for orbital magnetization calculations and other
        Berry-phase-related properties.
    
    **Returns:**
    
    function
        A JIT-compiled calculator function with signature:
        ```
        _calculator(k) -> (Hk, velocity_k)
        ```
        
        Parameters:
            k : jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            Hk : jnp.ndarray, shape (nwan, nwan)
                The Bloch Hamiltonian at k-point k.
            
            velocity_k : jnp.ndarray, shape (3, nwan, nwan)
                The velocity operator at k-point k.
                - velocity_k[0]: ∂H/∂k_x
                - velocity_k[1]: ∂H/∂k_y
                - velocity_k[2]: ∂H/∂k_z
                Each component is Hermitian.
    
    **Notes:**
    
    1. **Numerical Differentiation:**
       The velocity is computed using central finite differences with a step
       size of δ = 1e-6. This provides a good balance between accuracy and
       numerical stability. For higher precision, you may need to adjust δ.
    
    2. **Hermiticity:**
       Each component of the velocity operator is explicitly symmetrized as
       (v_μ + v_μ^†)/2 to ensure Hermiticity.
    
    3. **Anomalous Position Operator:**
       When `use_A=True`, the calculator internally creates an AA_k calculator
       using `make_calc_AA_k(model)`. This adds computational overhead but is
       necessary for correct orbital magnetization calculations.
    
    4. **Units:**
       - k is in units of 2π/a (reciprocal lattice vectors)
       - H(k) has units of energy (typically eV)
       - velocity_k has units of energy × length (eV·Å)
       - To get physical velocity, divide by ħ: v_physical = velocity_k / ħ
    
    5. **Performance:**
       The calculator is JIT-compiled with JAX. When `use_A=True`, the calculation
       involves additional matrix multiplications for the commutator term.
    
    6. **Gauge Invariance:**
       The velocity operator with the anomalous position term is gauge-invariant
       and correctly transforms under unitary transformations of the basis.
    
    
    """
    calc_Hk = make_k_hamiltonian_calculator(model)
    if use_A:
        calc_AA_k = make_calc_AA_k(model)
    delta = 1e-6

    @jax.jit
    def _calculator(k: jnp.ndarray):
        """
        Calculate the Hamiltonian and velocity operator at a k-point.
        
        This function computes the Bloch Hamiltonian H(k) and its derivative
        with respect to k (the velocity operator). If use_A=True, the velocity
        includes the anomalous position operator contribution.
        
        Args:
            k: jnp.ndarray, shape (3,)
                k-point in reciprocal space (in units of 2π/a).
        
        Returns:
            Hk: jnp.ndarray, shape (nwan, nwan)
                The Bloch Hamiltonian at k-point k.
            
            velocity_k: jnp.ndarray, shape (3, nwan, nwan)
                The velocity operator at k-point k.
                - velocity_k[0]: ∂H/∂k_x
                - velocity_k[1]: ∂H/∂k_y
                - velocity_k[2]: ∂H/∂k_z
                Each component is Hermitian.
        """
        # Calculate the Hamiltonian at the original k-point
        Hk = calc_Hk(k)
        
        # Calculate the velocity operator using central finite differences
        # ∂H/∂k_μ ≈ [H(k + δe_μ) - H(k - δe_μ)] / (2δ)
        # where e_μ is the unit vector in the μ direction
        
        # x-component: ∂H/∂k_x
        Hk_x_plus = calc_Hk(k + jnp.array([delta, 0.0, 0.0]))
        Hk_x_minus = calc_Hk(k - jnp.array([delta, 0.0, 0.0]))
        v_x = (Hk_x_plus - Hk_x_minus) / (2 * delta)
        
        # y-component: ∂H/∂k_y
        Hk_y_plus = calc_Hk(k + jnp.array([0.0, delta, 0.0]))
        Hk_y_minus = calc_Hk(k - jnp.array([0.0, delta, 0.0]))
        v_y = (Hk_y_plus - Hk_y_minus) / (2 * delta)
        
        # z-component: ∂H/∂k_z
        Hk_z_plus = calc_Hk(k + jnp.array([0.0, 0.0, delta]))
        Hk_z_minus = calc_Hk(k - jnp.array([0.0, 0.0, delta]))
        v_z = (Hk_z_plus - Hk_z_minus) / (2 * delta)
        
        # Stack the three components
        velocity_k = jnp.stack([v_x, v_y, v_z])
        
        # If use_A is True, include the anomalous position operator contribution
        # v_μ = ∂H/∂k_μ - i[A_μ, H]
        # where [A_μ, H] = A_μ H - H A_μ is the commutator
        if use_A:
            AA_k = calc_AA_k(k)
            # Calculate commutator for each component
            # commutator_μ = i * (AA_μ @ H - H @ AA_μ)
            commutator_x = 1j * (AA_k[:, :, 0] @ Hk - Hk @ AA_k[:, :, 0])
            commutator_y = 1j * (AA_k[:, :, 1] @ Hk - Hk @ AA_k[:, :, 1])
            commutator_z = 1j * (AA_k[:, :, 2] @ Hk - Hk @ AA_k[:, :, 2])
            
            # Subtract the commutator from the velocity
            velocity_k = velocity_k - jnp.stack([commutator_x, commutator_y, commutator_z])
        
        # Ensure each component of the velocity operator is Hermitian
        # This corrects for any numerical precision issues
        velocity_k = (velocity_k + velocity_k.transpose(0, 2, 1).conj()) / 2
        
        return Hk, velocity_k

    return _calculator


# Hk Calculators==================End