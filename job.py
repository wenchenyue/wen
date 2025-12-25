from TB_model import TB_model
from TB_io import read_TB_model
import operators as ops
from Hk import *
from calculators import *

model = read_TB_model("TB_Cu.npz")
#model.set_EF("/path/to/QE output")
# Or just set
model.EF = 12.43
model.set_E_fermis(-10,8,0.05)

matrix_all = ops.operator_matrix_all(['spd']) # 
Lz = matrix_all.pre_prepared_operators['Lz']
model.clear_calculator()
BC_calculator = model.make_calc_kubo(
                                "OHC_z",
                                alpha=0,
                                beta=1,
                                Gamma=0.025,
                                Operator=Lz)

model.make_calc_all(kbT=0.003,use_A=False)
model.make_k_mesh(80,80,80)

model.run_all("Cu",100,tabulate=False)
