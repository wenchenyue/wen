
import jax
import jax.numpy as jnp
from TB_model import TB_model

# using double precision
jax.config.update("jax_enable_x64", True)
# K space==============================BEGIN
def build_k_path(model:TB_model,
                 k_points:jnp.ndarray,
                 ticks_label:list[str],
                 length:int=100,
                 ):
    """
    build k path for given model
    param model: TB model
    param k_points: k points, shape (n_kpoints,3)
    param ticks_label: label for each tick, shape (n_kpoints,)
    param length: number of k points per path segment 
    it is a fix value so x coordinate of each k point is not evenly spaced
    return: k path, shape (n_kpoints,3)
    """
    # check k_points shape
    if k_points.shape[1] != 3:
        raise ValueError("k_points shape must be (n_kpoints,3)")
    n_path = k_points.shape[0] - 1
    if n_path < 1:
        raise ValueError("k_points must have at least 2 points")
    k_path = []   # all k points along path
    k_path_x = [] # x coordinate that used to plot path
    ticks = [] # xticks for plot
    X = 0
    for i in range(len(k_points)-1):
         k_start = k_points[i]
         k_end = k_points[i+1]
         interval = jnp.linalg.norm(k_end - k_start) / length # interval between two k points
         ticks.append(X)
         for j in jnp.arange(length):
             k_path.append(k_start + (k_end - k_start) * j / length)
             X += interval
             k_path_x.append(X)
    # the last point
    X += interval
    k_path_x.append(X)
    ticks.append(X)
    k_path.append(k_end)
    
    model.k_path = jnp.array(k_path)
    model.k_path_x = jnp.array(k_path_x)
    model.ticks = jnp.array(ticks)
    model.ticks_label = ticks_label

    return model.k_path,model.k_path_x,model.ticks,model.ticks_label

def build_k_mesh(model:TB_model,
                 nkx:int,
                 nky:int,
                 nkz:int,
                 ):
    """
    build k mesh for given model
    param model: TB model
    param nkx: k-mesh in x direction
    param nky: k-mesh in y direction
    param nkz: k-mesh in z direction
    return: k mesh, shape (nkx*nky*nkz,3)
    """
    nk = nkx * nky * nkz
    model.V_k = model.V/nk # normalize V to V_k

    k_mesh = jnp.meshgrid(jnp.linspace(0,1,nkx,endpoint=False),
                              jnp.linspace(0,1,nky,endpoint=False),
                              jnp.linspace(0,1,nkz,endpoint=False), 
                              indexing='ij')
    k_mesh = jnp.stack(k_mesh,axis=-1)
    k_points = k_mesh.reshape(-1,3)
    k_points = jnp.dot(k_points,model.recip_lattice)
    print(f"there are {nk} k points in k mesh")
    
    return k_mesh, k_points

#========================================END
