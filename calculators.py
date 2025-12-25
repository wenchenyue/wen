

import jax
import jax.numpy as jnp
from TB_model import TB_model
from utility import fermi_dirac_vamp
from Hk import make_k_hamiltonian_calculator, make_k_hamiltonian_velocity_calculator

# using double precision
jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64

levi_civita = jnp.array([
    [1,2],
    [2,0],
    [0,1]]
    )

# Calculators==================Begin


# one operator
def make_calc_inner(model: TB_model, Operator: jnp.ndarray):
    '''
    make a calculator for the inner product of an operator
    param:
        model: TB_model, # the model class
        Operator: jnp.ndarray, # the operator to calculate the inner product with, shape (nwan,nwan)
    return:
        _calculator: function, # a calculator for the inner product of the operator
    ---------------------------------------------------------------------------
    Tips:
    if operator O is a identity matrix, then the inner product represents the occupation number
    because <psi_n(k)|I|psi_m(k)> = <psi_n(k)|psi_m(k)> = delta_{n,m}
    after fermi scan, states below EF are occupied(1), and above EF are unoccupied(0)
    thus, the fermi scan results in the cumulative DOS, which is the integral of the DOS

    Similarly, if O is: diag([0,1,1,0,0,...]), where only some elements are 1, 
    then the PDOS (Partial Density of States for a specific orbital) can be calculated in this way.
    '''
    nwan = model.nwan
    assert Operator.shape == (nwan,nwan), f"Operator shape should be ({nwan},{nwan}), but got {Operator.shape}"

    @jax.jit
    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        '''
        calculate the inner product of the operator with the wave function psi_n(k)
        para:
            E_k: jnp.ndarray, # nwan, eigvals of Hk (ignored for inner product)
            Vec_k: jnp.ndarray, # nwan * nwan, eigvecs of Hk
            velocity_k: jnp.ndarray, # nwan * nwan * 3, velocity operator (ignored for inner product)
        return:
            inner_product: jnp.ndarray, # nwan
        '''
        # nwan
        inner_product = jnp.einsum('ij,jk,ki->i',Vec_k.conj().T,Operator,Vec_k, optimize='optimal')
        return inner_product.real
    return _calculator




# modern theory of orbital magnetism
def make_calc_mag_modern(model: TB_model,gamma:int = 2):
    '''
    make a calculator for the magnetism of the model
    using modern theory of orbital magnetism 
    Using Pezo's theory:

    [PHYSICAL REVIEW B 106, 104414 (2022)] by  Pezo
    this gives intrinsic moment,using formula:
    M_orb(n,k) = <u(n,k)|r x v|u(n,k)>
    M_orb(n,k) = Im <∂u(n,k)/∂k|x (H_k - E_nk)|∂u(n,k)/∂k>
               = 0.5 * Im sum_{q != n} 1/(E_qk-E_nk) <u_nk|vel_k|u_qk> X <u_qk|vel_k|u_nk>
    M_orb(n,k,gamma) = 
    Im sum_{q != n} 0.5/(E_qk-E_nk) <u_nk|vel_k[alpha]|u_qk><u_qk|vel_k[beta]|u_nk> - (alpha <-> beta)

    This result is the same as  the diagonal part of the magnetism matrix M_orb(n,k,gamma)

    param:
        model: TB_model, # the model data class
        gamma: int, # the direction of the magnetism, default is 2(z direction)
    return:
        _calculator: function, # a calculator for the  intrinsic magnetism of the model
    ---------------------------------------------------------------------------
    Formula:
    Note: velocity_k here defined with or without A term may result differently
    '''

    nwan = model.nwan
    print(f"modern theory of orbital magnetism, caution: only calculate the intrinsic part")
    alpha, beta = levi_civita[gamma]


    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        '''
        calculate the magnetism of the model
        para:
            E_k: jnp.ndarray, # nwan, eigvals of Hk
            Vec_k: jnp.ndarray, # nwan * nwan, eigvecs of Hk
            velocity_k: jnp.ndarray, # nwan * nwan * 3, velocity operator
        return:
            magnetism: jnp.ndarray, # nwan
        '''
        Vec_k_dagger = Vec_k.conj().T

        v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        v_beta = Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k
        
        Ediff = E_k[:,None] - E_k[None,:] # D[q,n] = E_qk - E_nk
        #Ediff = jnp.fill_diagonal(Ediff, jnp.inf, inplace=False) # D[n,n] = inf, to achieve n!=m in the SUM

        # the small imaginary part is to avoid division by zero
        Gqn = 1 / (Ediff+1e-12*1j) # 1/(E_qk-E_nk)

        # G[q,q] = 0, to achieve n!=m in the SUM    
        Gqn = jnp.fill_diagonal(Gqn, 0.0, inplace=False)
        
        # Cross product of v and v, in direction gamma: v_alpha * v_beta - (alpha <-> beta)
        # term1[n] = sum_{q} G[q,n] * v_alpha[n,q] * v_beta[q,n]
        cross_term1 = jnp.einsum("qn,nq,qn->n", Gqn, v_alpha, v_beta, optimize='optimal')
        # alternate version:
        # cross_term1 = jnp.sum(Gqn * v_alpha.T * v_beta, axis=0)

        # term2[n] = sum_{q} G[q,n] * v_beta[n,q] * v_alpha[q,n]
        cross_term2 = jnp.einsum("qn,nq,qn->n", Gqn, v_beta, v_alpha, optimize='optimal')
        # alternate version:
        #cross_term2 = jnp.sum(Gqn * v_beta.T * v_alpha, axis=0)
        
        # Take the imaginary part as specified in the formula
        magnetism = jnp.imag((cross_term1 - cross_term2) / 2)

        # using fact that cross_term2 is cross_term1.conj()
        # magnetism = jnp.imag(cross_term1)
        return magnetism 
    return _calculator

def make_calc_mag_matrix(model: TB_model, gamma:int = 2):
    """
    Make a calculator for the orbital magnetization matrix elements of the model.

    Args:
        model: TB_model
            The tight-binding model class instance.
        gamma: int, optional
            The direction index of the magnetization (0=x, 1=y, 2=z). Default is 2 (z direction).

    Returns:
        function: A JIT-compiled function `_calculator` that computes the magnetism matrix.

    References:
        Formula based on Eq. (6) in [PHYSICAL REVIEW B 106, 104414 (2022)] by Pezo et al.
        
        M_orb[n,m](k, gamma) = (1/4) * Im * Sum_{q != n, m} {
            (1 / (E_qk - E_nk)) * (v_alpha[n,q] * v_beta[q,m] - v_beta[n,q] * v_alpha[q,m]) +
            (1 / (E_qk - E_mk)) * (v_alpha[n,q] * v_beta[q,m] - v_beta[n,q] * v_alpha[q,m])
        }
        where alpha, beta are determined by the Levi-Civita symbol for the given gamma.
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
            E_k: jnp.ndarray
                Eigenvalues of H(k), shape (nwan,). real
            Vec_k: jnp.ndarray
                Eigenvectors of H(k), shape (nwan, nwan). Columns are eigenvectors. complex
            velocity_k: jnp.ndarray
                Velocity operator in the orbital basis, shape (nwan, nwan, 3). complex

        Returns:
            jnp.ndarray: The orbital magnetization matrix, shape (nwan, nwan).
        """
        Vec_k_dagger = Vec_k.conj().T
        # Transform velocity operators to the band basis: v_band = U^dagger * v_orb * U
        v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        v_beta = Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k

        # Calculate energy differences: Ediff[q, n] = E_q - E_n
        Ediff = E_k[:,None] - E_k[None,:] 
        # Set diagonal elements to infinity to avoid division by zero and effectively exclude q=n terms
        Ediff = jnp.fill_diagonal(Ediff, jnp.inf, inplace=False) 

        # G[q, n] = 1 / (E_q - E_n)
        G = 1.0/(Ediff+1e-12*1j) 

        # Part 1: Terms involving 1/(E_q - E_n)
        # Sum_q G[q,n] * (v_alpha[n,q] * v_beta[q,m] - v_beta[n,q] * v_alpha[q,m])
        term_G_qn = (
        jnp.einsum("qn,nq,qm->nm", G, v_alpha, v_beta) - 
        jnp.einsum("qn,nq,qm->nm", G, v_beta, v_alpha)
        )

        # Part 2: Terms involving 1/(E_q - E_m)
        # Sum_q G[q,m] * (v_alpha[n,q] * v_beta[q,m] - v_beta[n,q] * v_alpha[q,m])
        term_G_qm = (
        jnp.einsum("qm,nq,qm->nm", G, v_alpha, v_beta) - 
        jnp.einsum("qm,nq,qm->nm", G, v_beta, v_alpha) 
        )
        
        # Combine terms: M[n,m] = (1/4) * Im(Part1 + Part2)
        magnetism_matrix = (1/4) * jnp.imag(term_G_qn + term_G_qm)
        return magnetism_matrix
    return _calculator


#  a simpler version of kubo calculator
def make_calc_kubo(
    model: TB_model,
    alpha:int = 0,#1st direction of kubo formula
    beta:int  = 1,#2nd direction of kubo formula
    Gamma:float = 0.025,# smearing width,in eV unit
    Operator:jnp.ndarray = None, # default is None, then calc Berry Curvature
    ):
    """
    Make a calculator for the Kubo correlation function of the model.

    This calculator implements the Kubo formula for linear response calculations.
    It calculates the response function using the velocity operator and, optionally,
    an additional observable operator.

    If `Operator` is None, it calculates the standard Berry Curvature (intrinsic anomalous Hall conductivity).
    If `Operator` is provided, it calculates the Berry Curvature associated with that operator
    (e.g., spin Berry curvature for spin Hall conductivity), using the symmetrized current operator J_O = {O, v}/2.

    The calculation uses a Lorentzian broadening (smearing) parameter `Gamma` to handle energy denominators.

    Args:
        model: TB_model
            The tight-binding model class instance.
        alpha: int, optional
            The index of the first direction (response direction) in the Kubo formula (0=x, 1=y, 2=z). Default is 0.
        beta: int, optional
            The index of the second direction (perturbation direction) in the Kubo formula (0=x, 1=y, 2=z). Default is 1.
        Gamma: float, optional
            The smearing width (broadening parameter) in eV. Used in the energy denominator 1/((E_n-E_m)^2 + Gamma^2).
            Default is 0.025 eV.
        Operator: jnp.ndarray, optional
            An optional operator matrix in the orbital basis, shape (nwan, nwan).
            If provided, the response is calculated for the current operator J_O = 1/2 {Operator, v_alpha}.
            Default is None.

    Returns:
        function: A jittable function `_calculator` that computes the Kubo formula term for a given k-point.
    """
    nwan = model.nwan
    if Operator is None:
        print("calculate Berry Curvature")
    else:
        assert Operator.shape == (nwan,nwan), f"Operator shape {Operator.shape} is not (nwan,nwan)"

    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        """
        Calculate the Kubo formula term for a given k-point.

        Args:
            E_k: jnp.ndarray
                Eigenvalues of H(k), shape (nwan,).
            Vec_k: jnp.ndarray
                Eigenvectors of H(k), shape (nwan, nwan). Columns are eigenvectors.
            velocity_k: jnp.ndarray
                Velocity operator in the orbital basis, shape (nwan, nwan, 3).

        Returns:
            jnp.ndarray: The calculated Kubo term (e.g., Berry curvature) for each band, shape (nwan,).
        """
        Vec_k_dagger = Vec_k.conj().T
        
        # Transform the beta-component of velocity to the band basis: v_beta_band = U^dagger * v_beta_orb * U
        v_beta =  Vec_k_dagger @ velocity_k[:,:,beta] @ Vec_k
        # v_alpha will be calculated later depending on whether Operator is present

        # Calculate the energy denominator factor with Lorentzian broadening
        # denominator[n, m] = (E_n - E_m)^2 + Gamma^2
        denominator = (E_k[:,None] - E_k[None,:])**2 + Gamma**2 + 1e-12
        
        # G[n, m] = 1 / ((E_n - E_m)^2 + Gamma^2)
        # We exclude diagonal elements (n=m) by setting them to 0.0, as they don't contribute to interband transitions
        G = jnp.fill_diagonal(1.0/denominator, 0.0, inplace=False) 

        if Operator is None:
            # Pure Berry Curvature calculation
            # v_alpha_band = U^dagger * v_alpha_orb * U
            v_alpha = Vec_k_dagger @ velocity_k[:,:,alpha] @ Vec_k
        else:
            # Berry Curvature of an Operator (e.g., Spin Hall)
            # Construct the symmetrized current operator: J_O = 1/2 * {Operator, v_alpha}
            J_O = 0.5* (Operator @ velocity_k[:,:,alpha] + velocity_k[:,:,alpha] @ Operator)
            # Transform J_O to the band basis
            v_alpha =  Vec_k_dagger @ J_O @ Vec_k

        # Calculate the Kubo formula sum:
        # Omega_n = Sum_{m != n} G[n,m] * <n|v_alpha|m> * <m|v_beta|n>
        # 
        # Method 1: Using einsum (Compact and explicit)
        # "nm,nm,mn->n":
        #   G[n,m]      -> 1/((E_n-E_m)^2 + Gamma^2)
        #   v_alpha[n,m] -> <n|v_alpha|m>
        #   v_beta[m,n]  -> <m|v_beta|n>
        #   ->n          -> Sum over m for each n
        Omega_array = jnp.einsum("nm,nm,mn->n", G, v_alpha, v_beta) 
        
        # Method 2: Using element-wise multiplication and sum (Alternative, mathematically equivalent)
        # part_sum = G * v_alpha * v_beta.T  # v_beta.T[n,m] is v_beta[m,n]
        # Omega_array = jnp.sum(part_sum, axis=1)
        
        # The result is -2 * Im(Omega_array) to get the physical quantity (e.g., Berry curvature)
        return -2 * jnp.imag(Omega_array)
    return _calculator

def make_calc_all(model:TB_model,
                kbT:float=0.026, # room temperature
                use_A:bool=False,
                ):
    '''
    make a function to run a calculator for a list of k-points
    param model: TB_model, # the model class
    return: _calculator: function, # a function to run all calculator for a k-point
    '''        
    nwan = model.nwan
    all_calculators_name = list(model.all_calculators.keys())
    all_calculators_func = list(model.all_calculators.values())

    shape1 = len(all_calculators_name) + 1
    if len(all_calculators_name) == 0:
        print("no calculator found, calculate energy only")
        calculator_Hk = make_k_hamiltonian_calculator(model)
    else:
        print(f"calculate the following {shape1-1} calculators: {all_calculators_name}")
        calculator_Hk = make_k_hamiltonian_velocity_calculator(model,use_A)
    
    if kbT >1e-9:
        scan = True
        print(f"calculate the fermi scan with kbT = {kbT}")
        E_fermis = model.E_fermis
        shape2 = len(E_fermis)
    else: 
        scan = False
        print("no fermi scan, output all bands")


    #jit here
    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate all calculators for a k-point
        para:
            k: jnp.ndarray, # 3, k-point in reciprocal space
        return:
            results: jnp.ndarray, # shape (1+n_calculators, n_fermi(or n_band)), 
            results of all calculators
        '''
        # n_calc+1 * nwan
        result_bands = jnp.zeros((shape1, nwan), dtype=default_type_real)
        if shape1 == 1:
            Hk = calculator_Hk(k)
            E_k = jnp.linalg.eigvalsh(Hk) #nwan
            result_bands = result_bands.at[0].set(E_k)
        else:
            Hk, velocity_k = calculator_Hk(k)
            E_k,Vec_k = jnp.linalg.eigh(Hk)
            result_bands = result_bands.at[0].set(E_k)
            for i, func in enumerate(all_calculators_func):
                result_bands = result_bands.at[i+1].set(func(E_k,Vec_k,velocity_k))
        
        if not scan:
            return result_bands
        else:
            f_distribution = fermi_dirac_vamp(E_k,E_fermis,kbT) # nwan * len(E_fermis)
            result_scan = result_bands[:,:,None] * f_distribution[None,:,:] # n_calc+1 * nwan * len(E_fermis)
            return jnp.sum(result_scan,axis=1) # n_calc+1 * len(E_fermis),sum over nwan
    return _calculator

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

# test the calculator
# test pass
if __name__ == "__main__":
    from TB_model import TB_model
    from TB_io import read_TB_model
    import operators as ops
    from tqdm import trange

    model = read_TB_model("TB_Fe.npz")
    # set EF
    model.EF = 12.4712 #Cu:13.43  Fe:12.4712
    model.set_E_fermis(-0.6,0.6,0.01)

    mag_scan = make_calc_mag_scan(model,kbT=0.003,use_A=True,gamma=1)

    # add I calculator
    #cumDOS = make_calc_inner(model,Operator=jnp.eye(model.nwan))
    #model.add_calculator(cumDOS, "cumDOS")

    #ops_all = ops.operator_matrix_all(['spd'])
    #Lz = ops_all.pre_prepared_operators['Lz']
    #calc_Lz_OHC = make_calc_kubo(model,Gamma=0.003,Operator=Lz)
    #model.add_calculator(calc_Lz_OHC,"Lz_OHC")

    '''
    # add AHC calculator
    AHC_calculator = make_calc_kubo(model,Gamma=0.003,Operator=None)
    model.add_calculator(AHC_calculator,"AHC_z")
    AHC_calculator = make_calc_kubo(model,alpha=1,beta=2,Gamma=0.003,Operator=None)
    model.add_calculator(AHC_calculator,"AHC_x")
    AHC_calculator = make_calc_kubo(model,alpha=2,beta=0,Gamma=0.003,Operator=None)
    model.add_calculator(AHC_calculator,"AHC_y")


    # add morb calculator
    morb_calculator = make_calc_mag_modern(model,use_Pezo=False)
    model.add_calculator(morb_calculator,"morb_Lz")

    morb_calculator = make_calc_mag_modern(model,gamma=0,use_Pezo=False)
    model.add_calculator(morb_calculator,"morb_Lx")

    morb_calculator = make_calc_mag_modern(model,gamma=1,use_Pezo=False)
    model.add_calculator(morb_calculator,"morb_Ly")
    '''


    k_mesh,kpoints = model.make_k_mesh(150,150,150)
    nk = kpoints.shape[0]



    calculator_all =  mag_scan#make_calc_all(model,kbT=0.003,use_A=True)
    warm_up = calculator_all(jnp.array([0.0,0.0,0.0]))
    print(f"result shape: {warm_up.shape}")
    calculator_vmap = jax.jit(jax.vmap(calculator_all,in_axes=(0,)))

    results = jnp.zeros(warm_up.shape,device=jax.devices("cpu")[0],dtype=default_type_real)

    batch_size = 200
    batch_num = nk // batch_size

    for i in trange(batch_num):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_k = kpoints[start_idx:end_idx]
        result = calculator_vmap(batch_k) # batch_size * shape1 * shape2
        results += jnp.sum(result,axis=0) * model.V / nk
    
    jnp.save("results_Fe_mag_scan.npy",results)