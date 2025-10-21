# using numpy to store data, using jax to compute
import numpy as np
import jax
import jax.numpy as jnp
from utilty import cal_center_matrix, find_EF

# using double precision
jax.config.update("jax_enable_x64", True)


# data class========================= BEGIN
@dataclass
class TB_model:
    HR: jnp.ndarray = None # (nR ,nwan,nwan)
    cell_center: jnp.ndarray = None # (nR,3)
    real_lattice: jnp.ndarray = None # (3,3)
    recip_lattice: jnp.ndarray = None # (3,3)
    center_list: jnp.ndarray = None # (nwan,3)
    AA_R: jnp.ndarray = None # (nR,nwan,nwan,3)

    # post init: calculate some useful quantities
    def __post_init__(self):
        self.EF = 0.0

        self.nwan = self.HR.shape[-1] # number of wannier functions
        self.nR = self.HR.shape[0] # number of R points
        self.center_matrix = cal_center_matrix(self.center_list) # (nwan,nwan,3)

        b1 = self.recip_lattice[0]
        b2 = self.recip_lattice[1]
        b3 = self.recip_lattice[2]
        self.V = jnp.dot(b1,jnp.cross(b2,b3)) # volume of a reciprocal lattice cell
        
        # operator list
        self.O_list_inner: dict = {} # list of operators
        self.O_list_kubo: dict = {} # list of operator names    
        
        # print some info
        print (f"TB_model: nwan={self.nwan}, nR={self.nR}")
        
    
    def set_EF(self, file_name: str):
        self.EF = find_EF(file_name)
        print (f"EF={self.EF}")

    def set_E_fermis(self,
                      E_btm:float=-5,
                      E_top:float=5,
                      interval:float=0.05
                      ):  
        self.E_fermis = jnp.arange(self.EF+E_btm,self.EF+E_top,interval)
        print(f'set E_fermis from {self.EF+E_btm:.3f} to {self.EF+E_top:.3f}, interval {interval:.3f}, {len(self.E_fermis)} levels')
         

    def add_operator_kubo(self,operator: jnp.ndarray, name: str):
        '''
        add an operator to the model, and check its shape
        param O: operator matrix, shape (nwan,nwan)
        '''
        # check shape
        if operator.shape != (self.nwan,self.nwan):
            raise ValueError(f"operator {name} shape {operator.shape} not match nwan={self.nwan}")        
        self.O_list_kubo[name] = operator
        print (f"add operator {name} in kubo list")

    def add_operator_inner(self,operator: jnp.ndarray, name: str):
        '''
        add an operator to the model, and check its shape
        param O: operator matrix, shape (nwan,nwan)
        '''
        # check shape
        if operator.shape != (self.nwan,self.nwan):
            raise ValueError(f"operator {name} shape {operator.shape} not match nwan={self.nwan}")        
        self.O_list_inner[name] = operator
        print (f"add operator {name} in inner list")



    def clear_operator(self, category: str="all"):
        '''
        clear all operators in the model
        '''
        if category == "kubo":
            self.O_list_kubo = {}
            print (f"clear all operators in kubo list")
        elif category == "inner":
            self.O_list_inner = {}
            print (f"clear all operators in inner list")
        elif category == "all":
            self.O_list_kubo = {}
            self.O_list_inner = {}
            print (f"clear all operators in kubo list and inner list")
        else:
            raise ValueError(f"category {category} not supported")


