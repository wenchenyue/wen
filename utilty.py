# Use jax to compute
import jax
import jax.numpy as jnp

# using double precision
jax.config.update("jax_enable_x64", True)


#utilty ================================ BEGIN

# fermi-dirac funtion
def fermi_dirac(E,mu,kBT) -> float:
    return 1.0/(jnp.exp((E-mu)/kBT)+1)
# vamp it, E is an array, in  shape of (nE,), 
# return an F-D distrubution of a whole range of E
fermi_dirac_vamp = jax.vmap(fermi_dirac,in_axes=(0,None,None))

# calculate difference matrix
# only need to be computed once
def cal_center_matrix(center_list):
    '''
    param center_list: wannier center list, 2D array, shape (nwan,3)
    return: wannier center difference matrix, 3D array, shape (nwan,nwan,3)

    matrix[i,j] = vec[i] - vec[j]
    '''
    # check input
    assert center_list.ndim == 2, "center_list must be a 2D array"  
    assert center_list.shape[1] == 3, "center_list must have shape (nwan, 3)"
    # calculate wannier center difference matrix
    center_matrix = center_list[:, None, :] - center_list[None, :, :]
    return center_matrix

# find EF from .scf.out or .nscf.out file
def find_EF(file_name: str) -> float:
    '''
    param file_name: name of the .scf.out or .nscf.out file
    return: Fermi energy
    '''
    with open(file_name, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "the Fermi energy is" in line:
            EF = float(line.split()[-2])
            break
    return EF

#utilty ================================ END