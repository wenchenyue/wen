from tkinter import W
import numpy as np
import jax
import jax.numpy as jnp
from calculators import make_calc_inner, make_calc_kubo
from TB_model import TB_model
from calculators import make_k_hamiltonian_calculator



jax.config.update("jax_enable_x64", True)


def run_mesh(model:TB_model,
              kps:jnp.ndarray,
             calculator_inner:callable = None,
             calculator_kubo :callable = None,
             batch_size:int=1000,
             ):


    nk = kps.shape[0]
    # warm up
    if calculator_inner is None and calculator_kubo is None:
        raise ValueError("at least one of calculator_inner and calculator_kubo should be provided")
       
    # warm up
    warm_up1 = calculator_inner(jnp.array([0.0,0.0,0.0]))
    warm_up2 = calculator_kubo(jnp.array([0.0,0.0,0.0]))
    print(warm_up1.shape)
    print(warm_up2.shape)

def run_path():
    pass