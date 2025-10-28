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


def make_calc_inner(model: TB_model,kbT:float=0.0):
    '''
    make a calculator for the inner product of the operators
    param:
        model: TB_model, # the model class
        kbT: float, # the temperature in eV unit
    
    return:
        _calculator: function, # a calculator for the inner product of the operators
    
    This calculator calculates the inner product of the operators O with the wave function psi_n(k)
    in k space. The inner product is defined as:<psi_n(k)|O|psi_m(k)>
    if kbT< 1e-9, output  all of its states, otherwise output the fermi level scan result

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
        print("calculate only Energy")
    else:
        print(f"calculate the inner product of the operators: {model.O_list_inner.keys()}")

    # fermi scan
    if kbT > 1e-9:
        print(f"calculate the fermi scan with kbT = {kbT}")
        # check if E_fermis is set
        if model.E_fermis == None:
            raise ValueError("E_fermis not set, use set_E_fermis to set the Fermi energy")
        E_fermis = model.E_fermis
        E_min = E_fermis[0]
        E_max = E_fermis[-1]
        interval = E_fermis[1] - E_fermis[0]
        print(f"please confirm: E_min = {E_min}, E_max = {E_max}, interval = {interval}")
        scan = True
    else:
        # list of energy states, so E_fermis is not necessary
        print("do not calculate the fermi scan")
        scan = False

    hamiltonian_calculator = make_k_hamiltonian_calculator(model)

    # use jit
    # only k is dynamic, others are static
    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate the inner product of the operators
        para:
            k: jnp.ndarray, # 3
        return:
            inner_product: jnp.ndarray, physical values of one k-point
            it could be all of its states OR a fermi level scan result

        '''
        # calculate the hamiltonian in k space
        Hk = hamiltonian_calculator(k)
        if n_operators == 0:
            E_k = jnp.linalg.eigvalsh(Hk) # energy only
        else:
            E_k, Vec_k = jnp.linalg.eigh(Hk)
        
        if scan: 
            # a fermi distribution for each E_fermis[:]
            f_distribution = fermi_dirac_vamp(E_k,E_fermis,kbT) # nwan * len(E_fermis)
            result = jnp.zeros((n_operators+1,len(E_fermis)),dtype=default_type_real)
            temp = E_k[:,None] * f_distribution # nwan * len(E_fermis)
            result = result.at[0,:].set(jnp.sum(temp,axis=0)) # sum over nwan
            if n_operators > 0: # if O_list is not empty, this will do nothing 
                # use einsum to calculate the inner product of the operators
                # O_array is n_operators * nwan * nwan
                # Omega_array is n_operators * nwan
                # f_distribution is nwan * len(E_fermis)
                # the result is n_operators * len(E_fermis)
                Omega_array = jnp.einsum('ij,ajk,ki->ai',Vec_k.conj().T,O_array,Vec_k, optimize='optimal')
                temp = Omega_array.real[:,:,None] * f_distribution[None,:,:]
                result = result.at[1:,:].set(jnp.sum(temp,axis=1))
        else: 
            #output all of its states
            result = jnp.zeros((n_operators+1,nwan),dtype=default_type_real)
            result = result.at[0,:].set(E_k)
            if n_operators > 0: # if O_list is not empty, this will do nothing 
                #Omega_array is n_operators * nwan, take the real part
                Omega_array = jnp.einsum('ij,ajk,ki->ai',Vec_k.conj().T,O_array,Vec_k, optimize='optimal')
                result = result.at[1:,:].set(Omega_array.real)
        return result
    # end of _calculator

    
    return _calculator

def make_calc_kubo(
    model: TB_model,
    alpha:int = 0,#1st direction of kubo formula
    beta:int  = 1,#2nd direction of kubo formula
    Gamma:float = 0.025,# smearing width,in eV unit
    kbT:float=0.0, # in eV unit
    use_A:bool = False, # whether to use A_term, default is False
    ):
    '''
    make a calculator that calculates the kubo formula for the model
    para:
        model: TB_model, # the model to calculate
        alpha:int = 0,#1st direction of kubo formula
        beta:int  = 1,#2nd direction of kubo formula
        Gamma:float = 0.025,# smearing width,in eV unit
        kbT:float=0.0, # in eV unit
        use_A:bool = False, # whether to use A_term, default is False
    return:
        _calculator: a jitted function, # a calculator for the kubo formula
    '''

    O_array = jnp.array(list(model.O_list_kubo.values()))
    n_operators = O_array.shape[0]
    nwan = model.nwan

    if n_operators == 0:
        raise ValueError("O_list_kubo is empty, calculate nothing of kubo formula")
    else:
        print(f"calculate the kubo formula for the operators: {model.O_list_kubo.keys()}")
        Hk_Vk_calculator = make_k_hamiltonian_velocity_calculator(model,use_A)
    
    if kbT > 1e-9:
        print(f"calculate the fermi scan with kbT = {kbT}")
        # check if E_fermis is set
        if model.E_fermis == None:
            raise ValueError("E_fermis not set, use set_E_fermis to set the Fermi energy")
        E_fermis = model.E_fermis
        E_min = E_fermis[0]
        E_max = E_fermis[-1]
        interval = E_fermis[1] - E_fermis[0]
        print(f"please confirm: E_min = {E_min:.4f}, E_max = {E_max:.4f}, interval = {interval:.4f}")
        scan = True
    else:
        # list of energy states, so E_fermis is not necessary
        print("do not calculate the fermi scan")
        scan = False
    
    # An anonymous function that exists only within this function factory.
    def _cal_Omega(
        E_k:jnp.ndarray,
        Vec_k:jnp.ndarray,
        velocity_k:jnp.ndarray,
        ):
        '''
        calculate the Omega_k for the model
        para:
            E_k: jnp.ndarray, # nwan
            Vec_k: jnp.ndarray, # nwan * nwan
            velocity_k: jnp.ndarray, # nwan * nwan
        return:
            Omega_array: jnp.ndarray, # n_operators * nwan
        '''
        Vec_k_dagger = Vec_k.conj().T
        v_beta = velocity_k[:,:,beta]
        v_alpha = velocity_k[:,:,alpha]

        #term 2 for the kubo formula, it is O-independent
        v_beta_eb = Vec_k_dagger @ v_beta @ Vec_k

        # denominator for the kubo formula
        denominator = (E_k[None,:] - E_k[:,None])**2 + Gamma**2 +1e-12 # nwan * nwan
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
        return Omega_array
            

    # jit the function
    @jax.jit
    def _calculator(
        k:jnp.ndarray,
        ):
        '''
        calculate the Omega_k for the model
        para:
            k: jnp.ndarray, # 3
        return:
            Omega_array: jnp.ndarray, # n_operators * nwan
        '''
        H_k, velocity_k = Hk_Vk_calculator(k)
        E_k,Vec_k = jnp.linalg.eigh(H_k)
        if scan:
                f_distribution = fermi_dirac_vamp(E_k,E_fermis,kbT) # nwan * len(E_fermis)
                Omega_array = _cal_Omega(E_k,Vec_k,velocity_k)
                temp = Omega_array.real[:,:,None] * f_distribution[None,:,:] # n_operators * nwan * len(E_fermis)
                # n_operators * len(E_fermis)
                result = jnp.sum(temp,axis=1)
        else:
            result = _cal_Omega(E_k,Vec_k,velocity_k)
        return result
    # end of _calculator
    return _calculator
        
# Calculators==================End

