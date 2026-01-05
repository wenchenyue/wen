# Use jax to compute
import jax
import jax.numpy as jnp
from TB_model import TB_model
from utility import DEFAULT_TYPE_COMPLEX, DEFAULT_TYPE_REAL

# using double precision
# jax.config.update("jax_enable_x64", True)

# io ================================ BEGIN
def read_TB_model(file_name: str) -> TB_model:
    """
    Read a TB model from a npz file.

    Parameters
    ----------
    file_name : str
        The name of the npz or pickle file.

    Returns
    -------
    TB_model
        The TB model.
    """

    # check file extension
    if file_name.endswith(".npz"):
        print (f"read_TB_model: read {file_name}")
        data = jnp.load(file_name)
        HR = data['HR'].astype(DEFAULT_TYPE_COMPLEX)
        cell_center = data['cRvec'].astype(DEFAULT_TYPE_REAL)
        center_list = data['Wan_center'].astype(DEFAULT_TYPE_REAL)
        AA_R = data['AA_R'].astype(DEFAULT_TYPE_COMPLEX)
        real_lattice = data['real_lattice'].astype(DEFAULT_TYPE_REAL)
        recip_lattice = data['recip_lattice'].astype(DEFAULT_TYPE_REAL)
        
        return TB_model(HR=HR,
                        cell_center=cell_center,
                        center_list=center_list,
                        AA_R=AA_R,
                        real_lattice=real_lattice,
                        recip_lattice=recip_lattice)
    elif file_name.endswith(".pickle"):
        import pickle
        with open(file_name, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        raise ValueError(f"Unknown file extension {file_name}")


def save_TB_model(model: TB_model, file_name: str):
    """
    Save a TB model to a pickle file.

    Parameters
    ----------
    model : TB_model
        The TB model to be saved.
    file_name : str
        The name of the pickle file.
    """
    import pickle
    with open(file_name+".pickle", "wb") as f:
        pickle.dump(model, f)
# io ================================ END