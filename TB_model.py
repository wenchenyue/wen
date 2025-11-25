# using numpy to store data, using jax to compute
import numpy as np
import jax
import jax.numpy as jnp
from utility import cal_center_matrix, find_EF
#data class========================= BEGIN
from dataclasses import dataclass


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
    k_points: jnp.ndarray = None # (nk,3)

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
        # calculator
        self.inner_calculator = None
        self.kubo_calculator = None
        
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
    

    def make_k_hamiltonian_calculator(self):
        '''
        make a calculator for k-pt Hamiltonian
        return: a function, which takes k as input, and return H(k) as output
        '''
        from calculators import make_k_hamiltonian_calculator
        return make_k_hamiltonian_calculator(self)
    
    def make_k_hamiltonian_velocity_calculator(self):
        '''
        make a calculator for k-pt Hamiltonian velocity
        return: a function, which takes k as input, and return H(k) as output
        '''
        from calculators import make_k_hamiltonian_velocity_calculator
        return make_k_hamiltonian_velocity_calculator(self)
    
    def make_calc_inner(self,kbT:float=0.01):
        '''
        make a calculator for inner quantity
        return: a function, which takes k as input, and return inner quantity as output
        '''
        from calculators import make_calc_inner
        self.inner_calculator = make_calc_inner(self)
        return self.inner_calculator    
    
    def make_calc_kubo(self,
                        alpha:int = 0,#1st direction of kubo formula
                        beta:int  = 1,#2nd direction of kubo formula
                        Gamma:float = 0.025,# smearing width,in eV unit
                       ):
        '''
        make a calculator for kubo quantity
        return: a function, which takes k as input, and return kubo quantity as output
        '''
        from calculators import make_calc_kubo
        self.kubo_calculator = make_calc_kubo(self,alpha,beta,Gamma)
        return self.kubo_calculator 
    
    def make_calc_all(self,
                    alpha:int=0,
                    beta:int=1,
                    Gamma:float=0.025,
                    kbT:float=0.026, # room temperature
                    use_A:bool=False,
                        ):
        '''
        make all calculators for given model
        return: a dictionary of calculators
        '''
        from calculators import make_calc_all
        self.all_calculator = make_calc_all(self,alpha,beta,Gamma,kbT,use_A)
        return self.all_calculator
    
    def make_k_mesh(self,
                    nkx:int=100,
                    nky:int=100,
                    nkz:int=100,
                    ):
        '''
        make a k-mesh for calculation
        return: a k-mesh, shape (nkx,nky,nkz,3)
        '''
        from K_space import build_k_mesh
        kmesh, k_points = build_k_mesh(self,nkx,nky,nkz)
        self.k_mesh = kmesh
        self.k_points = k_points
        self.nkx = nkx
        self.nky = nky
        self.nkz = nkz
        return kmesh, k_points
    
    def make_k_path(self,
                    k_points:jnp.ndarray,
                    ticks_label:list[str],
                    length:int=100,
                    ):
        '''
        build k path for given model
        param model: TB model
        param k_points: k points, shape (n_kpoints,3)
        param ticks_label: label for each tick, shape (n_kpoints,)
        param length: number of k points per path segment 
        it is a fix value so x coordinate of each k point is not evenly spaced
        return: k path, shape (n_kpoints,3)
        '''
        from K_space import build_k_path
        return build_k_path(self,k_points,ticks_label,length)
        
    def run_all(self,batch_size:int=100,tabulate:bool=False):
        '''
        run all calculators for given k points
        '''

        res_dict = {}
        if len(self.O_list_inner) > 0:
            res_dict['inner'] = {}
        if len(self.O_list_kubo) > 0:
            res_dict['kubo'] = {}
        
        if self.k_points is None:
            raise ValueError("k_points is None, please make k mesh first")
        k_points = self.k_points
        nk = k_points.shape[0]
        V_k =self.V / nk

        from run import run_kps
        res = run_kps(k_points,self.all_calculator,batch_size=batch_size,tabulate=tabulate)
        print(res.shape)
        res_dict['energy'] = res[0]
        i = 1
        for key in self.O_list_inner.keys():
            if tabulate:
                tmp = res[:,i,:]
                res_dict['inner'][key] = tmp.reshape([self.nkx,self.nky,self.nkz,-1])
            else:
                tmp = res[i,:]
                res_dict['inner'][key] = tmp*V_k
            i += 1
        for key in self.O_list_kubo.keys():
            if tabulate:
                tmp = res[:,i,:]
                res_dict['kubo'][key] = tmp.reshape([self.nkx,self.nky,self.nkz,-1])
            else:
                tmp = res[i,:]
                res_dict['kubo'][key] = tmp*V_k
            i += 1
                
        # time
        import time
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print(f"finish kubo calculation at {time_str}")
        print("------------------------")

        # save to pickle file
        import pickle
        with open(f"res_dict_{time_str}.pkl", "wb") as f:
            pickle.dump(res_dict, f)


# test pass
if __name__ == "__main__":
    from TB_io import read_TB_model
    import operators as ops
    model = read_TB_model("TB.npz")
    # fermi level
    model.EF = 13.43#model.set_EF("Cu.scf.out")
    model.set_E_fermis(-10,8,0.05)
    # operators
    model.add_operator_inner(jnp.eye(model.nwan), "cumDOS")
    #use ops
    matrix_all = ops.operator_matrix_all(['spd'])
    model.add_operator_kubo(matrix_all.pre_prepared_operators["Lz"], "Lz")

    #calculators
    #_inner_calculator = model.make_calc_inner()
    #_kubo_calculator =  model.make_calc_kubo()
    _all_calculator = model.make_calc_all(kbT=0.026)

    # k_mesh
    _,_ = model.make_k_mesh(nkx=100,nky=100,nkz=100)

    # run 
    model.run_all(batch_size=1000,tabulate=False)