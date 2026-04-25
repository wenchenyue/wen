from TB_model import TB_model
from TB_io import read_TB_model
import operators as ops
from Hk import *
from calculators import *

jax.config.update("jax_enable_x64", True)

model = read_TB_model("TB_Cu.npz")
#model.set_EF("/path/to/QE output")
# Or just set
model.EF = 12.43
model.set_E_fermis(-10,8,0.05)

matrix_all = ops.operator_matrix_all(['spd']) # 
Lz = matrix_all.pre_prepared_operators['Lz']
model.clear_calculator()
model.make_calc_kubo(
                    "OHC_z",
                    alpha=0,
                    beta=1,
                    Gamma=0.025,
                    Operator=Lz,
                    H_gauge=False,
                    )

Lz_modern = make_calc_mag_matrix(model,2)
model.make_calc_kubo(
                    "modern",
                    alpha=0,
                    beta=1,
                    Gamma=0.025,
                    Operator=Lz_modern,
                    H_gauge=True
                    )




model.make_calc_all(kbT=0.0,use_A=False)
model.make_k_mesh(40,40,40)

model.run_all("Cu",800,tabulate=True,frmsf=True)
