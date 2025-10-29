import operator
from tqdm import trange
import numpy as np
import jax
import jax.numpy as jnp

from TB_model import TB_model


jax.config.update("jax_enable_x64", True)
default_type_complex = jnp.complex128
default_type_real = jnp.float64


def run_kps(
              kps:jnp.ndarray,
             calculator:callable,
             batch_size:int=1000,
             tabulate:bool=False,
             ):


    nk = kps.shape[0]
       
    # warm up
    warm_up = calculator(jnp.array([0.0,0.0,0.0]))
    print(f"result shape: {warm_up.shape}")
    calculator_vmap = jax.vmap(calculator,in_axes=(0,))

    if tabulate:
        # nk, n_operator, n_fermi(or n_band)
        results = jnp.zeros([nk,warm_up.shape[0],warm_up.shape[1]],
                             device=jax.devices("cpu")[0],dtype=default_type_real)
        # estimate memory usage
        mem_usage = nk * warm_up.size * 8 / 1024 / 1024
        print(f"Use tabulating, estimated memory usage: {mem_usage:.3f} MB")
    else:# n_operator, n_fermi(or n_band)
        results = jnp.zeros(warm_up.shape,device=jax.devices("cpu")[0],dtype=default_type_real)

    # check GPU
    
    if jax.local_devices()[0].platform == 'gpu':
        gpu_num = len(jax.local_devices())
        calculator_pmap = jax.pmap(calculator_vmap)
        print(f"GPU is available, number: {gpu_num}")  
    else:
        gpu_num = 0
        print("GPU is not available, use CPU")


    batch_num = nk // batch_size
    #   check batch_size
    if (gpu_num > 0) and (batch_size % gpu_num != 0):
        raise ValueError(f"batch_size {batch_size} should be divisible by gpu_num {gpu_num}")
    if nk % batch_size != 0:
        raise ValueError(f"nk {nk} should be divisible by batch_size {batch_size}")
    
    # hint about batch_size
    print(f"{batch_num} batchs,{batch_size} k-points per batch")
    print("if out of memory, try to decrease batch_size")

    # run
    for i in trange(batch_num):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_k = kps[start_idx:end_idx]
        
        if gpu_num > 0:
            batch_k = batch_k.reshape([gpu_num,batch_size//gpu_num,3])
            result = calculator_pmap(batch_k)
            # gpu_num, batch_size//gpu_num, n_operator, n_fermi(or n_band)
            # to batch_size, n_operator, n_fermi(or n_band)
            result = result.reshape([batch_size,*result.shape[2:]])
        else:
            result = calculator_vmap(batch_k)
        
        if tabulate:
            results = results.at[start_idx:end_idx].set(result)
        else:
            results += jnp.sum(result,axis=0)

    return results


def run_all(model:TB_model,
            kbT:float = 0.026,
            batch_size:int=1000,
            tabulating:bool = False,
            use_A = False,
            ):
    inner_calculator = model.make_calc_inner(kbT = 0.026)
    if len(list(model.O_list_inner.values()))>0:
        kubo_calculator = model.make_calc_kubo(kbT = 0.026,)


if __name__ == "__main__":
    from TB_model import TB_model
    from TB_io import read_TB_model
    import operators as ops

    model = read_TB_model("TB.npz")
    # set EF
    model.set_EF("Cu.scf.out")
    model.set_E_fermis(-10,8,0.05)
    # add identity operator
    model.add_operator_inner(jnp.eye(model.nwan), "I")

    # use operator "I" to create Kubo operator
    matrix_all = ops.operator_matrix_all(['spd'])
    
    model.add_operator_kubo(matrix_all.pre_prepared_operators["Lz"], "Lz")

    inner_calculator = model.make_calc_inner(kbT = 0.026)
    kubo_calculator =  model.make_calc_kubo(kbT=0.026)

    # make k-mesh
    k_mesh_flatten = model.make_k_mesh(100,100,100)
    print(model.k_mesh.shape)
    print(k_mesh_flatten.shape)
    # run

    res_kubo = run_kps(k_mesh_flatten,
                       kubo_calculator,
                       batch_size=1000,
                       tabulate=False)
    print(res_kubo.shape)
    jnp.save("kubo_scan.npy",res_kubo)





