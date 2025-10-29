import jax
import jax.numpy as jnp
from matplotlib import use


# using double precision
jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64


num_ao = { # num of atomic orbitals for each orbital type, considering spin
    's':2,
    'p':6,
    'd':10,
    'f':14, # no g orbital
}
# load pre-defined operator matrix
operator = jnp.load("operator.npz")

class operator_matrix_all:
    def __init__(self,orbital_list:list[str],use_LS_base=False):
        self.orbital_list = orbital_list
        self.orbital_all = build_operator_all(orbital_list)
        self.ndim = len(self.orbital_all)

        self.S,self.L,self.LS = build_operator_LS(orbital_list,use_LS_base=use_LS_base)

        self.pre_prepared_operators = {
            'Sx' :self.S[:,:,0],
            'Sy' :self.S[:,:,1],
            'Sz' :self.S[:,:,2],
            'Lx' :self.L[:,:,0],
            'Ly' :self.L[:,:,1],
            'Lz' :self.L[:,:,2],
            'LS' :self.LS,
            "I"  :jnp.eye(self.ndim,dtype=default_type_real),
        }


    
    def make_P_easy(self,select_name:str):
        """
        make projection operator for given orbital name

        param select_name: name of the orbital, e.g. "s" or "p"
        return: projection operator, shape (norb,norb)
        """
        return build_P_easy(self.orbital_all,select_name)
    
    def make_POP_easy(self,select_name:str,pre_perpared_name:str):
        """
        make projection operator for given orbital name

        param select_name: name of the orbital, e.g. "s" or "p"
        param pre_perpared_name: name of the pre-prepared operator, e.g. "Sx" or "Ly"
        return: projected operator, P O P ,shape (norb,norb)
        """
        return build_P_easy(self.orbital_all,select_name) @ self.pre_prepared_operators[pre_perpared_name] @ build_P_easy(self.orbital_all,select_name) 

    def make_POP_pre_prepared(self,P_mat:jnp.ndarray,pre_perpared_name:str):
        """
        make projection operator for given orbital name

        param P_mat: projection operator, shape (norb,norb)
        param pre_perpared_name: name of the pre-prepared operator, e.g. "Sx" or "Ly"
        return: projected operator, P O P ,shape (norb,norb)
        """
        return P_mat @ self.pre_prepared_operators[pre_perpared_name] @ P_mat


# Function to build projection operator for given orbital list
def build_P_easy(orbital_all:list[str],selected:str):
    """
    build projection operator for given orbital list

    param orbital_list: list of orbital names, e.g. ["spd","p"]
    return: projection operator, shape (norb,norb)
    """
    select_list = [1 if o==selected else 0 for o in orbital_all]    
    return jnp.diag(jnp.array(select_list,dtype=default_type_real))


def build_operator_all(orbital_list:list[str]):
    """
    build operator name list for given orbital list

    param orbital_list: list of orbital names, e.g. ["spd","p"]
    return: operator name list repeat for each orbital type, 
    e.g. ["s","s","p","p","p","d","d","d"]
    """
    orbital_str = "".join(orbital_list)
    orbital_all = []
    for o in orbital_str:
        if o not in num_ao:
            raise ValueError(f"orbital type {o} not supported")
        num_repeat = num_ao[o]
        orbital_all.extend([o]*num_repeat)
 
    return orbital_all

def select_orbital_easy(orbital_name_list:list[str],selected:str):
    """
    select orbital (1 or 0) for given orbital name list

    param orbital_name_list: list of orbital names, e.g. ["s","s","p","p","p","d","d","d"]
    param orbital_name: name of the orbital, e.g. "s" or "p"
    return: same size list of 1 or 0, e.g. [1,1,0,0,0,1,1,1]
    """
    return [1 if o==selected else 0 for o in orbital_name_list]

def build_operator_LS(orbital_list:list[str],use_LS_base:bool=False):
    """
    build operator matrix for given orbital list

    param orbital_list: list of orbital names, e.g. ["spd","p","sd",'f']
    param use_LS_base: whether to use LS coupled base, 
    default False, using real Atomic orbital basis
    return: S,L in 3D array, shape (norb,norb,3) and LS in 2D array, shape (norb,norb)
    """
    orbital_str = "".join(orbital_list)

    orbital_all = []
    for o in orbital_str:
        if o not in num_ao:
            raise ValueError(f"orbital type {o} not supported")
        orbital_all.append(num_ao[o])
    
    # init operator matrix
    norb = sum(orbital_all)
    S = jnp.zeros((norb,norb,3),dtype=default_type_complex)
    L = jnp.zeros((norb,norb,3),dtype=default_type_complex)

    # choose which operator matrix to use
    if use_LS_base:
        S_all = operator["S_LS"]
        L_all = operator["L_LS"]
    else:
        S_all = operator["S"]
        L_all = operator["L"]

    # seperate each orbitals into a dictionary
    S_AO = {
        2:S_all[0:2,0:2,:], # 2
        6:S_all[2:8,2:8,:], # 6
        10:S_all[8:18,8:18,:], # 10
        14:S_all[18:32,18:32,:], # 14    
    }
    L_AO = {
        2:L_all[0:2,0:2,:],# 2
        6:L_all[2:8,2:8,:],# 6
        10:L_all[8:18,8:18,:],# 10  
        14:L_all[18:32,18:32,:],# 14
    }  

    # fill L and S matrix
    i = 0
    for n_ao in orbital_all:
        start = i
        end = i + n_ao
        S = S.at[start:end,start:end,:].set(S_AO[n_ao])
        L = L.at[start:end,start:end,:].set(L_AO[n_ao])
        i = end
    
    # L dot S
    LS = L[:,:,0] @ S[:,:,0] + L[:,:,1] @ S[:,:,1] + L[:,:,2] @ S[:,:,2]

    return S,L,LS


# test
if __name__ == "__main__":
    orbital_list = ["spd"]
    S,L,LS = build_operator_LS(orbital_list,use_LS_base=True)
    test1 = operator_matrix_all(['spd'])
    print(test1.orbital_all)
    print(test1.pre_prepared_operators.keys())
    print(test1.pre_prepared_operators["LS"].shape)

    O_test_P = test1.make_P_easy("s")
    print(jnp.diag(O_test_P))
