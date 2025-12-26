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
    This is vital, reason from [read more].
    
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
    
    Read more:
    wanniertools -- theory --- tight-binding model
     T. Yusufaly, D. Vanderbilt, and S. Coh, Tight-Binding Formalism in the Context of the PythTB Package, (n.d.).
     web1: https://www.wanniertools.org/theory/tight-binding-model/
     web2: https://pythtb.readthedocs.io/en/latest/formalism.html

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
    

    formula from equation (39) of [PHYSICAL REVIEW B 74, 195118 (2006)]

    I add phase_W to the formula.
    
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
    
    **Hermiticity:**
       Each component of AA_k is explicitly symmetrized as (AA_k + AA_k^†)/2
       to ensure Hermiticity.



    
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
def make_k_hamiltonian_velocity_calculator(model: TB_model,use_A:bool=False):
    """
    Factory function to create a Hamiltonian velocity calculator.

    This function generates a JIT-compiled calculator that computes both the k-space Hamiltonian
    and its velocity derivatives (∂H/∂k) for a tight-binding model.

    Args:
        model: TB_model instance containing tight-binding parameters
        use_A: bool, if True include anomalous position term from orbital Hall effect

    Returns:
        A JIT-compiled function that takes k (3D vector) and returns:
            Hk: jnp.ndarray, shape (nwan, nwan) - k-space Hamiltonian
            velocity_k: jnp.ndarray, shape (nwan, nwan, 3) - velocity derivatives
    """
    cell_center = model.cell_center
    center_matrix = model.center_matrix
    HR = model.HR
    if use_A:
        print("Anomalous position term is used")
        AA_k_calc = make_calc_AA_k(model)
    else:
        print("Anomalous position term is not used,use partial only")

    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        Calculate the velocity of the Hamiltonian with respect to k.

        Args:
            k: jnp.ndarray, # 3

        Returns:
            Hk: jnp.ndarray, # nwan * nwan
            velocity_k: jnp.ndarray, # nwan * nwan * 3
        '''
        # Calculate phase factors for lattice and sublattice centers
        dot_product = jnp.einsum('ij,j->i', cell_center, k)
        Phase_R = jnp.exp(1j * dot_product) # nR
        dot_product = jnp.einsum('ijk,k->ij', center_matrix, k)
        Phase_W = jnp.exp(1j * dot_product) # nwan * nwan

        # Compute Hamiltonian in k-space via Wannier interpolation
        # H_{ij}(k) = Σ_R e^{ik·R} H_{ij}(R) e^{ik·(τ_j - τ_i)}
        # where τ_j - τ_i are the 'center_matrix'
        partial_sum = jnp.einsum('i,ijk->jk', Phase_R, HR) # nwan * nwan
        Hk = partial_sum * Phase_W
        Hk = (Hk + Hk.conj().T) / 2

        # Derivative of H(k) with respect to k: ∂H(k)/∂k_μ
        # ∂H(k)/∂k_μ = i Σ_R e^{ik·(R+τ_ji)} (R_μ + τ_ji,μ) H(R)
        # where τ_ji,μ = τ_j,μ - τ_i,μ is the μ-component of the center difference
        part1 = jnp.einsum('iab, i, ab, id -> abd',
                            HR,
                            Phase_R,
                            Phase_W,
                            cell_center
                            )
        part2 = jnp.einsum('iab, i, ab, abd -> abd',
                            HR,
                            Phase_R,
                            Phase_W,
                            center_matrix)
        Hk_derivative = 1j*(part1 + part2)
        Hk_derivative = (Hk_derivative + Hk_derivative.transpose(1,0,2).conj()) / 2

        if use_A:
            # Add anomalous position contribution from orbital Hall effect
            # Ref: D. Go, Phys. Rev. B 109, 174435 (2024)
            # eq. (31)
            # The anomalous velocity term is:
            # A_{mn}(k) = -i ( <R|A|R>_{mn} - <R|A|R>_{nm}^* )
            # This accounts for the orbital-dependent part of the anomalous position operator
            AA_k = AA_k_calc(k)
            A_term = -1j * (jnp.einsum('ijl,jk->ikl',AA_k,Hk) - jnp.einsum('ij,jkl->ikl',Hk,AA_k))
            velocity_k = Hk_derivative + A_term
        else:
            velocity_k = Hk_derivative
        return Hk,velocity_k
    return _calculator

# Hk Calculators==================End