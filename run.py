import numpy as np
import jax
import jax.numpy as jnp
from calculators import make_calc_inner, make_calc_kubo

jax.config.update("jax_enable_x64", True)


def run_mesh(kps:jnp.ndarray,
             calculator:callable,
             batch_size:int=1000,
             ):
    # warm up
    wram_up = my_calculator(jnp.array([0.0,0.0,0.0]))
    print(wram_up.shape)

def run_path():
    pass