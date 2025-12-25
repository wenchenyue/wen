import jax
import jax.numpy as jnp
from TB_model import TB_model



# using double precision
jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64

# Hk Calculators==================Begin

#hamiltonian in k space
# use functional factory to make a calculator
# pass a dataclass because the calculator should be coupled to the model

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


# may be useful to calculate the AA_k in  Wannier gauge
def make_calc_AA_k(model: TB_model):
    cell_center = model.cell_center
    center_matrix = model.center_matrix
    AA_R = model.AA_R

    @jax.jit
    def _calculator(k:jnp.ndarray):
        '''
        calculate the anomalous position operator in k space(wannier gauge)
        para:
            k: jnp.ndarray, # 3
        return:
            AA_k: jnp.ndarray, # nwan * nwan * 3
        '''
        # calculate the phase factor for each R
        # This phase is caused by the diffrent position of the cell center
        dot_product = jnp.einsum('ij,j->i', cell_center, k)
        Phase_R = jnp.exp(1j * dot_product) # nR
        partial_sum = jnp.einsum('i,ijkl->jkl', Phase_R, AA_R)
        
        dot_product = jnp.einsum('ijk,k->ij', center_matrix, k)
        Phase_W = jnp.exp(1j * dot_product) # nwan * nwan
        

        AA_k = partial_sum * Phase_W[:,:,None]
        AA_k = (AA_k + AA_k.transpose(1, 0, 2).conj()) / 2 # make sure the AA_k is hermitian
        return AA_k
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
        AA_k_calc = make_calc_AA_k(model)
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
        Hk_derivative = (Hk_derivative + Hk_derivative.transpose(1,0,2).conj()) / 2
        if use_A:
            # A_{mn}(k) = -i ( <R|A|R>_{mn} - <R|A|R>_{nm}^* )
            AA_k = AA_k_calc(k)
            A_term = -1j * (jnp.einsum('ijl,jk->ikl',AA_k,Hk) - jnp.einsum('ij,jkl->ikl',Hk,AA_k))
            velocity_k = Hk_derivative + A_term
        else:
            velocity_k = Hk_derivative
        return Hk,velocity_k    
    return _calculator

# Hk Calculators==================End