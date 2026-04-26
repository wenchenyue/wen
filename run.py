
from tqdm import trange
import jax
import numpy as np
# print jax version
print(jax.__version__)
import jax.numpy as jnp
from utility import DEFAULT_TYPE_COMPLEX, DEFAULT_TYPE_REAL


# single gpu version
def run_kps_gpu_single(
             kps:jnp.ndarray,
             calculator:callable,
             batch_size:int=1000,
             tabulate:bool=False,
             ):
    '''
    run a calculator for a list of k-points on single GPU
    param kps: k-points, shape (nk,3)
    param calculator: a function, which takes k as input, and return observable as output
    param batch_size: batch size, default is 1000
    param tabulate: whether to tabulate the results, default is False
    return: observable, shape (n_operator,nk,n_fermi(or n_band)) if tabulate is True, else (n_operator,n_fermi(or n_band))
    '''
    gpu_devices = jax.devices('gpu')
    gpu_device = gpu_devices[0]

    nk = kps.shape[0]

    # warm up
    warm_up = calculator(jnp.array([0.0,0.0,0.0]))
    warm_shape = warm_up.shape
    assert len(warm_shape) == 2, "calculator should return a 2D array"

    calculator_vmap = jax.vmap(calculator,in_axes=(0,))

    if tabulate:
        mem_usage = nk * warm_up.nbytes / 1024 / 1024
        print(f"Use tabulating, estimated memory usage: {mem_usage:.3f} MB")
        results_list = []
    else:
        Results = jnp.zeros(warm_shape, device=gpu_device, dtype=DEFAULT_TYPE_REAL)

    batch_num = nk // batch_size
    if nk % batch_size != 0:
        raise ValueError(f"nk {nk} should be divisible by batch_size {batch_size}")

    print(f"{batch_num} batchs,{batch_size} k-points per batch")
    print("if out of memory, try to decrease batch_size")

    batch_k_set = kps.reshape([batch_num, batch_size, 3])

    for i in trange(batch_num):
        batch_k = batch_k_set[i]
        batch_k = jax.device_put(batch_k, device=gpu_device)

        result = calculator_vmap(batch_k)

        if tabulate:
            result = jax.device_put(result, device=jax.devices("cpu")[0])
            results_list.append(result)
        else:
            result = jnp.sum(result, axis=0)
            Results += jax.device_put(result, device=gpu_device)

    if tabulate:
        result = np.concatenate(results_list, axis=0)
        return np.transpose(result, (1, 0, 2))
    return Results

# multiple gpu version
def run_kps_gpu_multiple(
             kps:jnp.ndarray,
             calculator:callable,
             batch_size:int=1000,
             tabulate:bool=False,
             ):
    '''
    run a calculator for a list of k-points on multiple GPUs
    param kps: k-points, shape (nk,3)
    param calculator: a function, which takes k as input, and return observable as output
    param batch_size: batch size, default is 1000
    param tabulate: whether to tabulate the results, default is False
    return: observable, shape (n_operator,nk,n_fermi(or n_band)) if tabulate is True, else (n_operator,n_fermi(or n_band))
    '''
    gpu_devices = jax.devices('gpu')
    gpu_num = len(gpu_devices)
    gpu_device = gpu_devices[0]

    nk = kps.shape[0]

    # warm up
    warm_up = calculator(jnp.array([0.0,0.0,0.0]))
    warm_shape = warm_up.shape
    assert len(warm_shape) == 2, "calculator should return a 2D array"

    calculator_vmap = jax.vmap(calculator,in_axes=(0,))
    calculator_pmap = jax.pmap(calculator_vmap)

    if tabulate:
        mem_usage = nk * warm_up.nbytes / 1024 / 1024
        print(f"Use tabulating, estimated memory usage: {mem_usage:.3f} MB")
        results_list = []
    else:
        Results = jnp.zeros(warm_shape, device=gpu_device, dtype=DEFAULT_TYPE_REAL)

    batch_num = nk // batch_size
    if batch_size % gpu_num != 0:
        raise ValueError(f"batch_size {batch_size} should be divisible by gpu_num {gpu_num}")
    if nk % batch_size != 0:
        raise ValueError(f"nk {nk} should be divisible by batch_size {batch_size}")

    print(f"{batch_num} batchs,{batch_size} k-points per batch")
    print("if out of memory, try to decrease batch_size")

    batch_k_set = kps.reshape([batch_num, batch_size, 3])

    for i in trange(batch_num):
        batch_k = batch_k_set[i]

        batch_k = batch_k.reshape([gpu_num, batch_size // gpu_num, 3])
        batch_k = jax.device_put_sharded(list(batch_k), devices=gpu_devices)
        result = calculator_pmap(batch_k)
        result = result.reshape([batch_size, warm_shape[0], warm_shape[1]])

        if tabulate:
            result = jax.device_put(result, device=jax.devices("cpu")[0])
            results_list.append(result)
        else:
            result = jnp.sum(result, axis=0)
            Results += jax.device_put(result, device=gpu_device)

    if tabulate:
        result = np.concatenate(results_list, axis=0)
        return np.transpose(result, (1, 0, 2))
    return Results

def run_kps_cpu(
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
    return: observable, shape (n_operator,nk,n_fermi(or n_band)) if tabulate is True, else (n_operator,n_fermi(or n_band))
    '''
    # make sure using cpu device
    cpu_device = jax.devices("cpu")[0]
    jax.default_device(cpu_device)
    
    nk = kps.shape[0]
       
    # warm up
    warm_up = calculator(jnp.array([0.0,0.0,0.0]))
    warm_shape = warm_up.shape
    assert len(warm_shape) == 2, "calculator should return a 2D array"

    calculator_vmap = jax.vmap(calculator,in_axes=(0,))

    if tabulate:
        mem_usage = nk * warm_up.nbytes / 1024 / 1024
        print(f"Use tabulating, estimated memory usage: {mem_usage:.3f} MB")
        results_list = []
    else:
        Results = jnp.zeros(warm_shape, device=cpu_device, dtype=DEFAULT_TYPE_REAL)

    batch_num = nk // batch_size
    if nk % batch_size != 0:
        raise ValueError(f"nk {nk} should be divisible by batch_size {batch_size}")

    print(f"{batch_num} batchs,{batch_size} k-points per batch")
    batch_k_set = kps.reshape([batch_num, batch_size, 3])

    for i in trange(batch_num):
        batch_k = batch_k_set[i]

        result = calculator_vmap(batch_k)

        if tabulate:
            results_list.append(np.asarray(result))
        else:
            result = jnp.sum(result, axis=0)
            Results += result

    if tabulate:
        result = np.concatenate(results_list, axis=0)
        return np.transpose(result, (1, 0, 2))
    return Results

# run
def run_kps(
             kps:jnp.ndarray,
             calculator:callable,
             batch_size:int=1000,
             tabulate:bool=False,
             ):
    """
    Router function that automatically selects GPU or CPU for k-point calculations.
    
    This function detects available devices and routes the computation to either
    run_kps_gpu_single, run_kps_gpu_multiple or run_kps_cpu accordingly.
    
    Parameters
    ----------
    kps : jnp.ndarray
        k-points array, shape (nk, 3).
    calculator : callable
        Calculator function with signature calculator(k) -> jnp.ndarray.
    batch_size : int, optional
        Number of k-points per batch. Default is 1000.
    tabulate : bool, optional
        Whether to tabulate all results. Default is False.
    
    Returns
    -------
    jnp.ndarray
        Calculation results. See run_kps_gpu_single, run_kps_gpu_multiple or run_kps_cpu for details.
    """
    # check if GPU available
    if jax.local_devices()[0].platform == 'gpu':
        gpu_devices = jax.devices('gpu')
        gpu_num = len(gpu_devices)
        if gpu_num > 1:
            print(f"use multiple gpus: {gpu_num}")
            return run_kps_gpu_multiple(kps,calculator,batch_size,tabulate)
        else:
            print("use single gpu")
            return run_kps_gpu_single(kps,calculator,batch_size,tabulate)
    else:
        print("use cpu")
        return run_kps_cpu(kps,calculator,batch_size,tabulate)

 




# test pass
if __name__ == "__main__":
    pass





