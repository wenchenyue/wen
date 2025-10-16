# using numpy to store data, using jax to compute
import numpy as np
import jax
import jax.numpy as jnp

# pre-defined orbitals
orbital_list = ['s','p','d','f']

# using double precision
jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64

@dataclass
class TB_model:
    HR: np.ndarray = None # (nR ,nwan,nwan)
    cell_center: np.ndarray = None # (nR,3)
    real_lattice: np.ndarray = None # (3,3)
    recip_lattice: np.ndarray = None # (3,3)
    center_list: np.ndarray = None # (nwan,3)
    AA_R: np.ndarray = None # (nR,nwan,nwan,3)

    def __post_init__(self):
        self.nwan = self.HR.shape[-1]
        