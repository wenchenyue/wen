import re
import jax
import jax.numpy as jnp
from TB_model import TB_model
from utility import fermi_dirac_vamp


# using double precision
jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64


# Calculators==================Begin

#hamiltonian in k space
# use functional factory to make a calculator
# pass a dataclass because the calculator should be coupled to the model

from typing import Any


def make_k_hamiltonian_calculator(model: TB_model):
    cell_center = model.cell_center
    center_matrix = model.center_matrix
    HR = model.HR

    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate the hamiltonian in k space
        para:
            k: jnp.ndarray, # 3
        return:
            Hk: jnp.ndarray, # nwan * nwan
        '''
        # calculate the phase factor for each R
        # This phase is caused by the diffrent position of the cell center
        dot_product = jnp.einsum('ij,j->i', cell_center, k)
        Phase_R = jnp.exp(1j * dot_product) # nR

        # calculate the phase factor for each center diff
        # This phase is caused by the diffrent position of the WF within the unit cell
        dot_product = jnp.einsum('ijk,k->ij', center_matrix, k)
        Phase_W = jnp.exp(1j * dot_product) # nwan * nwan
        
        # part 1: use einsum to calculate the partial sum of Phase_R and HR
        partial_sum = jnp.einsum('i,ijk->jk', Phase_R, HR)
        # part 2: multiply the phase factor for each center diff
        Hk = partial_sum * Phase_W

        # make sure the hamiltonian is hermitian
        return (Hk + Hk.conj().T) / 2

    return _calculator


# a calculator for the velocity of the hamiltonian with respect to k
# it should be coupled, so just pass the model class
def make_k_hamiltonian_velocity_calculator(model: TB_model,use_A:bool=False):
    cell_center = model.cell_center
    center_matrix = model.center_matrix
    HR = model.HR
    AA_R = model.AA_R
    nwan = center_matrix.shape[0]

    if use_A:
        print("Anomalous position term is used")
    else:
        print("Anomalous position term is not used,use partial only")

    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate the velocity of the hamiltonian with respect to k
        para:
            k: jnp.ndarray, # 3
        return:
            Hk: jnp.ndarray, # nwan * nwan
            velocity_k: jnp.ndarray, # nwan * nwan * 3
        '''
        # calculate 2 phase factors
        dot_product = jnp.einsum('ij,j->i', cell_center, k)
        Phase_R = jnp.exp(1j * dot_product) # nR
        dot_product = jnp.einsum('ijk,k->ij', center_matrix, k)
        Phase_W = jnp.exp(1j * dot_product) # nwan * nwan

        # Hk hamiltonian in k space
        # part 1: use einsum to calculate the partial sum of Phase_R and HR
        partial_sum = jnp.einsum('i,ijk->jk', Phase_R, HR) # nwan * nwan   
        Hk = partial_sum * Phase_W
        Hk = (Hk + Hk.conj().T) / 2

        # There are 2 parts of derivative of Hk with respect to k
        # ∂H(k)/∂k_μ = i Σ_R e^{ik·(R+τ_ij)} (R_μ+τ_ij,μ) H(R)
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

        if use_A:
            # A_{mn}(k) = -i ( <R|A|R>_{mn} - <R|A|R>_{nm}^* )
            partial_sum = jnp.einsum('i,ijkl->jkl', Phase_R, AA_R)
            temp = (partial_sum + partial_sum.transpose(1, 0, 2).conj()) / 2
            A_term = -1j * (jnp.einsum('ijl,jk->ikl',temp,Hk) - jnp.einsum('ij,jkl->ikl',Hk,temp))
            velocity_k = Hk_derivative + A_term
        else:
            velocity_k = Hk_derivative
        return Hk,velocity_k    
    return _calculator


def make_calc_inner(model: TB_model):
    '''
    make a calculator for the inner product of the operators
    param:
        model: TB_model, # the model class
    return:
        _calculator: function, # a calculator for the inner product of the operators
    
    This calculator calculates the inner product of the operators O with the wave function psi_n(k)
    in k space. The inner product is defined as:<psi_n(k)|O|psi_m(k)>
    
    Tips:
    if operator O is a identity matrix, then the inner product represents the occupation number
    because <psi_n(k)|I|psi_m(k)> = <psi_n(k)|psi_m(k)> = delta_{n,m}
    after fermi scan, states below EF are occupied(1), and above EF are unoccupied(0)
    thus, the fermi scan results in the cumulative DOS, which is the integral of the DOS

    Similarly, if O is: diag([0,1,1,0,0,...]), where only some elements are 1, 
    then the PDOS (Partial Density of States for a specific orbital) can be calculated in this way.
    '''

    O_array = jnp.array(list(model.O_list_inner.values()))
    n_operators = O_array.shape[0]
    nwan = model.nwan
    if n_operators == 0:
        raise ValueError("O_list_inner is empty, please add some operators")
    else:
        print(f"calculate the inner product of the operators: {model.O_list_inner.keys()}")
    
    def _calculator(Vec_k:jnp.ndarray):
        '''
        calculate the inner product of the operators
        para:
            E_k: jnp.ndarray, # nwan
            Vec_k: jnp.ndarray, # nwan * nwan
        return:
            inner_product: jnp.ndarray, # n_operators * nwan
        '''
        # n_operators * nwan
        inner_product = jnp.einsum('ij,ajk,ki->ai',Vec_k.conj().T,O_array,Vec_k, optimize='optimal')
        return inner_product
    return _calculator






def make_calc_kubo(
    model: TB_model,
    alpha:int = 0,#1st direction of kubo formula
    beta:int  = 1,#2nd direction of kubo formula
    Gamma:float = 0.025,# smearing width,in eV unit
    ):
    '''
    make a calculator for the Kubo formula
    param:
        model: TB_model, # the model to calculate
        alpha:int = 0,#1st direction of kubo formula
        beta:int  = 1,#2nd direction of kubo formula
        Gamma:float = 0.025,# smearing width,in eV unit
    return:
        _calculator: function, # a calculator for the kubo formula
    
    This calculator implements the Kubo formula for conductivity calculations.
    It calculates the response function using the velocity operator and the provided operators.
    The Kubo formula is implemented with a smearing parameter Gamma to avoid divergence.
    '''

    O_array = jnp.array(list(model.O_list_kubo.values()))
    n_operators = O_array.shape[0]
    nwan = model.nwan

    if  n_operators == 0:
        raise ValueError("O_list_kubo is empty, calculate nothing of kubo formula")
    else:
        print(f"calculate the kubo formula for the operators: {model.O_list_kubo.keys()}")

    def _calculator(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        '''
        calculate the kubo formula for the model
        para:
            E_k: jnp.ndarray, # nwan, eigvals of Hk,
            Vec_k: jnp.ndarray, # nwan * nwan, eigvecs of Hk
            velocity_k: jnp.ndarray, # nwan * nwan, velocity of each state in k space
        return:
            kubo: jnp.ndarray, # n_operators * nwan
        
        The calculation proceeds in several steps:
        1. Extract the velocity components in alpha and beta directions
        2. Transform velocity operator to eigenbasis
        3. Compute the off-diagonal terms with smearing to avoid divergence
        4. Sum over all transitions to get the response for each operator
        '''
        # calculate the hamiltonian in k space
        Vec_k_dagger = Vec_k.conj().T
        v_beta = velocity_k[:,:,beta]
        v_alpha = velocity_k[:,:,alpha]

        #term 2 for the kubo formula, it is O-independent
        v_beta_eb = Vec_k_dagger @ v_beta @ Vec_k 

        # denominator for the kubo formula
        denominator = (E_k[None,:] - E_k[:,None])**2 + Gamma**2 + 1e-12 # nwan * nwan
        # set to inf where n=m
        diag_indices = jnp.diag_indices(nwan)
        denominator = denominator.at[diag_indices].set(jnp.inf)

        # here we can use for loop, because n_operators is small
        Omega_array = jnp.zeros((n_operators,nwan),dtype=default_type_real)
        for i,O in enumerate(O_array):
            J_O = (0.5) * (O @ v_alpha + v_alpha @ O)
            # term1 is O dependent
            J_O_eb = Vec_k_dagger @ J_O @ Vec_k  # term1
            numerator = J_O_eb * v_beta_eb.T # term1 * term2: O[n,m] * v_beta[m,n]
            fraction = numerator / denominator # nwan * nwan
            # sum over n, and take the imaginary part
            Omega_array = Omega_array.at[i,:].set(-2 * jnp.imag(jnp.sum(fraction,axis=1))) 
        return Omega_array # n_operators * nwan
    return _calculator


        
def make_calc_all(model:TB_model,
                alpha:int=0,
                beta:int=1,
                Gamma:float=0.025,
                kbT:float=0.026, # room temperature
                use_A:bool=False,
                ):
    '''
    make a function to run a calculator for a list of k-points
    param model: TB_model, # the model class
    return: run_kps: function, # a function to run a calculator for a k-point
    '''
    # make Hk calculators
    nwan = model.nwan
    calculator_Hk = make_k_hamiltonian_velocity_calculator(model,use_A)

    shape1 = 1 # result shape for Hk
    # make inner calculators
    O_array_inner = jnp.array(list(model.O_list_inner.values()))
    n_operators_inner = O_array_inner.shape[0]
    if n_operators_inner != 0:
        calculator_inner = make_calc_inner(model)
        shape1 += n_operators_inner
    else:
        calculator_inner = None

    # make kubo calculators
    O_array_kubo = jnp.array(list(model.O_list_kubo.values()))
    n_operators_kubo = O_array_kubo.shape[0]
    if n_operators_kubo != 0:
        calculator_kubo = make_calc_kubo(model,alpha,beta,Gamma)
        shape1 += n_operators_kubo
    else:
        calculator_kubo = None

    if kbT >1e-9:
        print(f"calculate the fermi scan with kbT = {kbT}")
        # check if E_fermis is set
        if model.E_fermis == None:
            raise ValueError("E_fermis not set, use set_E_fermis to set the Fermi energy")
        E_fermis = model.E_fermis
        scan = True
        shape2 = len(E_fermis)
    else:
        shape2 = nwan
        scan = False

    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate the observable for a k-point
        param k: k-point, shape (3,)
        return: observable, shape (1+n_operators_inner+n_operators_kubo,n_fermi(or n_band))
        '''
        Hk,velocity_k= calculator_Hk(k)
        E_k,Vec_k = jnp.linalg.eigh(Hk)
        result = jnp.zeros((shape1,shape2),dtype=default_type_real)
        if scan:
            f_distribution = fermi_dirac_vamp(E_k,E_fermis,kbT) # nwan * len(E_fermis)
            temp = E_k[:,None] * f_distribution # nwan * len(E_fermis)
            result = result.at[0,:].set(jnp.sum(temp,axis=0))# sum over nwan

            if calculator_inner != None:
                inner_array = calculator_inner(Vec_k) # n_operators_inner * nwan
                temp = inner_array.real[:,:,None] * f_distribution[None,:,:] # n_operators_inner * nwan * len(E_fermis)
                result = result.at[1:1+n_operators_inner,:].set(jnp.sum(temp,axis=1)) # sum over nwan
            
            if calculator_kubo != None:
                kubo_array = calculator_kubo(E_k,Vec_k,velocity_k) # n_operators_kubo * nwan
                temp = kubo_array.real[:,:,None] * f_distribution[None,:,:] # n_operators_kubo * nwan * len(E_fermis)
                result = result.at[1+n_operators_inner:,:].set(jnp.sum(temp,axis=1)) # sum over nwan
        
        else:
            result = result.at[0,:].set(E_k) # nwan
            if calculator_inner != None:
                inner_array = calculator_inner(Vec_k).real # n_operators_inner * nwan
                result = result.at[1:1+n_operators_inner,:].set(inner_array) # set the inner array
            if calculator_kubo != None:
                kubo_array = calculator_kubo(E_k,Vec_k,velocity_k).real # n_operators_kubo * nwan
                result = result.at[1+n_operators_inner:,:].set(kubo_array) # set the kubo array
        
        return result
    return _calculator

# Calculators==================End

# test the calculator
# test pass
if __name__ == "__main__":
    from TB_model import TB_model
    from TB_io import read_TB_model
    import operators as ops
    from tqdm import trange

    model = read_TB_model("TB.npz")
    # set EF
    model.EF = 13.43
    model.set_E_fermis(-10,8,0.05)
    # add identity operator
    model.add_operator_inner(jnp.eye(model.nwan), "I")
    matrix_all = ops.operator_matrix_all(['spd'])
    model.add_operator_kubo(matrix_all.pre_prepared_operators["Lz"], "Lz")
    # make k-mesh
    k_mesh,kpoints = model.make_k_mesh(80,80,80)
    nk = kpoints.shape[0]



    calculator_all = make_calc_all(model)
    warm_up = calculator_all(jnp.array([0.0,0.0,0.0]))
    print(f"result shape: {warm_up.shape}")
    calculator_vmap = jax.vmap(calculator_all,in_axes=(0,))

    results = jnp.zeros(warm_up.shape,device=jax.devices("cpu")[0],dtype=default_type_real)

    batch_size = 100
    batch_num = nk // batch_size

    for i in trange(batch_num):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_k = kpoints[start_idx:end_idx]
        result = calculator_vmap(batch_k) # batch_size * shape1 * shape2
        results += jnp.sum(result,axis=0)
    
    jnp.save("results.npy",results)