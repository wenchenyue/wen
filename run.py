
from tqdm import trange
import jax
import jax.numpy as jnp




jax.config.update("jax_enable_x64", True)
DEFAULT_TYPE_COMPLEX = jnp.complex128
DEFAULT_TYPE_REAL = jnp.float64

def run_kps(
             kps:jnp.ndarray,
             calculator:callable,
             batch_size:int=1000,
             tabulate:bool=False,
             ):
    '''
    run a calculator for a list of k-points
    param kps: k-points, shape (nk,3)
    param calculator: a function, which takes k as input, and return observable as output
    param batch_size: batch size, default is 1000
    param tabulate: whether to tabulate the results, default is False
    return: observable, shape (nk,n_operator,n_fermi(or n_band)) if tabulate is True, else (n_operator,n_fermi(or n_band))
    '''


    # estimate size of kps, use nbyte
    kps_size = kps.nbytes
    print(f"kps size: {kps_size / 1024 / 1024:.3f} MB")

    nk = kps.shape[0]
       
    # warm up
    warm_up = calculator(jnp.array([0.0,0.0,0.0]))
    print(f"result shape: {warm_up.shape}")
    calculator_vmap = jax.vmap(calculator,in_axes=(0,))

    if tabulate:
        # nk, 1 + n_calc_all, n_fermi(or n_band)
        Results = jnp.zeros([nk,warm_up.shape[0],warm_up.shape[1]],
                             device=jax.devices("cpu")[0],dtype=DEFAULT_TYPE_REAL)
        # estimate memory usage
        mem_usage = nk * warm_up.size * 8 / 1024 / 1024
        print(f"Use tabulating, estimated memory usage: {mem_usage:.3f} MB")
    else:# 1 + n_calc_all, n_fermi(or n_band)
        Results = jnp.zeros(warm_up.shape,device=jax.devices("gpu")[0],dtype=DEFAULT_TYPE_REAL)

    # check GPU
    if jax.local_devices()[0].platform == 'gpu':
        gpu_devices = jax.devices('gpu')
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

    batch_k_set = kps.reshape([batch_num,batch_size,3])

    # run
    for i in trange(batch_num):
        # get batch k and put to gpu if available
        batch_k = batch_k_set[i]
        if gpu_num > 0:
            batch_k = jax.device_put(batch_k, device=gpu_devices[0])
        
        ## running step
        if gpu_num > 1:
            batch_k = batch_k.reshape([gpu_num,batch_size//gpu_num,3])
            batch_k = jax.device_put_sharded(list(batch_k), devices=gpu_devices)
            result = calculator_pmap(batch_k)
            # gpu_num, batch_size//gpu_num, n_operator, n_fermi(or n_band)
            # to batch_size, n_operator, n_fermi(or n_band)
            result = result.reshape([batch_size,*result.shape[2:]])

        else:
            result = calculator_vmap(batch_k)
        
        ## post processing
        if tabulate:
            # 直接取回结果用于切片赋值
            result = jax.device_put(result, device=jax.devices("cpu")[0])
            Results = Results.at[i*batch_size:(i+1)*batch_size].set(result)
        else:
            # 在GPU上先求和，再取回CPU，避免在CPU上进行求和操作
            result = jnp.sum(result, axis=0)
            result = jax.device_put(result, device=jax.devices("gpu")[0])
            Results += result

    return Results



# test pass
if __name__ == "__main__":
    from TB_model import TB_model
    from TB_io import read_TB_model
    import operators as ops
    from calculators import *
    model = read_TB_model("TB.npz")
    nwan = model.nwan

    
    model.EF = 12.471
    model.set_E_fermis(-1.6,1.6,0.01)
    # add identity operator

    mag_scan = make_calc_mag_scan(model,kbT=0.003,use_A=True)
    #cumDOS = make_calc_inner(model,Operator=jnp.eye(model.nwan))
    #model.add_calculator(cumDOS,'cumDOS')



    # use operator "I" to create Kubo operator
    matrix_all = ops.operator_matrix_all(['spd'])
    
    Sz = matrix_all.pre_prepared_operators['Sz']
    
    
    model.add_calculator(calc_Sz,"Sz")
    #calc_Lz_OHC = make_calc_kubo(model,Gamma=0.003,Operator=Lz)
    #model.add_calculator(calc_Lz_OHC,'Lz_OHC')

    #inner_calculator = model.make_calc_inner(kbT=0.026)
    #kubo_calculator =  model.make_calc_kubo(kbT=0.026)
    all_calculator = mag_scan # make_calc_all(model,kbT=0.0003,use_A=True)

    # make k-mesh
    k_mesh,k_mesh_flatten = model.make_k_mesh(160,160,160)
    print(model.k_mesh.shape)
    print(k_mesh_flatten.shape)
    # run

    nk = k_mesh_flatten.shape[0]
    res = run_kps(k_mesh_flatten,
                       all_calculator,
                       batch_size=2000,
                       tabulate=False)
    print(res.shape)
    jnp.save("test_noSym.npy",res * model.V / nk) 





