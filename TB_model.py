# using numpy to store data, using jax to compute
import numpy as np
import jax
import jax.numpy as jnp
from utility import cal_center_matrix, find_EF,DEFAULT_TYPE_COMPLEX, DEFAULT_TYPE_REAL
from dataclasses import dataclass

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
        self.E_fermis = None
        self.nwan = self.HR.shape[-1] # number of wannier functions
        self.nR = self.HR.shape[0] # number of R points
        self.center_matrix = cal_center_matrix(self.center_list) # (nwan,nwan,3)
        self.distance_all = self.get_distance_all() # in shape (nR,nwan,nwan)

        b1 = self.recip_lattice[0]
        b2 = self.recip_lattice[1]
        b3 = self.recip_lattice[2]
        self.V = jnp.dot(b1,jnp.cross(b2,b3)) # volume of a reciprocal lattice cell
        
        # operator list
        self.all_calculators:dict = {}
        self.O_list_inner: dict = {} # list of operators
        self.O_list_kubo: dict = {} # list of operator names   
        # calculator
        self.inner_calculator = None
        self.kubo_calculator = None
        
        # print some info
        print (f"TB_model: nwan={self.nwan}, nR={self.nR}")


    def get_distance_all(self):
        """
        calculate the distance between all pairs of orbitals
        Delta(m,n,R) = (rn + R) - rm
        """
        diff_all = jnp.zeros((self.nR,self.nwan,self.nwan,3), dtype=DEFAULT_TYPE_REAL)
        for i, vec in enumerate(self.cell_center):
            diff_all = diff_all.at[i,:,:,:].set(self.center_matrix + vec)
        distance_all = jnp.linalg.norm(diff_all, axis=-1)
        return distance_all     
    
    def cut_off(self, cut_off_radius: float = 6.0):
        """
        cut off the Hamiltonian matrix
        drop all the matrix elements whose distance is larger than cut_off_radius
        """

        print(f"cut off the Hamiltonian elements with distance > {cut_off_radius}")
        
        # in shape (nR,nwan,nwan)
        mask = self.distance_all <= cut_off_radius
        self.HR = self.HR * mask 

        exclude_list = []   
        for i in range(self.nR):
            H_max = jnp.max(jnp.abs(self.HR[i,:,:]))
            if H_max < 1e-9:
                exclude_list.append(i)
        exclude_array = jnp.array(exclude_list)
        self.HR = jnp.delete(self.HR, exclude_array, axis=0)
        self.cell_center = jnp.delete(self.cell_center, exclude_array, axis=0)
        self.distance_all = jnp.delete(self.distance_all, exclude_array, axis=0)
        self.AA_R = jnp.delete(self.AA_R, exclude_array, axis=0)
        self.nR = self.HR.shape[0]


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
         
    def add_calculator(self, calculator: callable, name:str):
        self.all_calculators[name] = calculator
    
    def clear_calculator(self):
        '''
        clear all calculators in the model
        '''
        self.all_calculators = {}
   

    def make_k_hamiltonian_calculator(self):
        '''
        make a calculator for k-pt Hamiltonian
        return: a function, which takes k as input, and return H(k) as output
        '''
        from calculators import make_k_hamiltonian_calculator
        return make_k_hamiltonian_calculator(self)
    
    def make_k_hamiltonian_velocity_calculator(self,use_A:bool=False):
        '''
        make a calculator for k-pt Hamiltonian velocity
        return: a function, which takes k as input, and return H(k) as output
        '''
        from calculators import make_k_hamiltonian_velocity_calculator
        return make_k_hamiltonian_velocity_calculator(self,use_A)
    
    def make_calc_inner(self,name:str,operator=None, H_gauge:bool=False):
        '''
        make a calculator for inner quantity
        return: a function, which takes Vec_k as input, and return inner quantity as output
        '''
        from calculators import make_calc_inner
        self.all_calculators[name] = make_calc_inner(self,operator, H_gauge)
        return self.all_calculators[name]    
    
    def make_calc_intra(self,name:str,beta:int=2, operator=None,H_gauge:bool=False):
        '''
        make a calculator for intra quantity
        return: a function, which takes Vec_k and Vel_k as input, and return intra quantity as output
        '''
        from calculators import make_calc_intra
        self.all_calculators[name] = make_calc_intra(self,beta,operator, H_gauge)
        return self.all_calculators[name]  


    def make_calc_kubo(self,
                        name:str,
                        alpha:int = 0,#1st direction of kubo formula
                        beta:int  = 1,#2nd direction of kubo formula
                        Gamma:float = 0.025,# smearing width,in eV unit
                        Operator= None, # default is None, then calc Berry Curvature
                        H_gauge:bool=False, # whether the operator is in the Hamiltonian(band) basis
                       ):
        '''
        make a calculator for kubo quantity
        return: a function, which takes E_k, Vec_k, Vel_k as input, 
        and return kubo quantity as output
        '''
        from calculators import make_calc_kubo
        self.all_calculators[name] = make_calc_kubo(self,alpha,beta,Gamma,Operator,H_gauge)
        return self.all_calculators[name] 
    
    def make_calc_OBC(self,
                        name:str,
                        alpha:int = 0,#1st direction of kubo formula
                        beta:int  = 1,#2nd direction of kubo formula
                        gamma:int = 2,#direction of Orbital magnetization moment
                        Gamma:float = 0.025,# smearing width,in eV unit, default is 0.025
                        H_gauge:bool=False, # whether the operator is in the Hamiltonian(band) basis
                        ):
        '''
        make a calculator for OBC modern version
        return: a function, which takes E_k, Vec_k, Vel_k as input, and return OBC as output
        '''
        #warn
        print("Warning: OBC is using  modern version, using make_calc_mag_matrix to calculate Lz_modern")

        from calculators import make_calc_kubo,make_calc_mag_matrix
        Lz_modern = make_calc_mag_matrix(self,gamma)
        self.all_calculators[name] = make_calc_kubo(self,alpha,beta,Gamma,Lz_modern,H_gauge)
        return self.all_calculators[name]

    def make_calc_all(self,
                    kbT:float=0.026, # room temperature
                    use_A:bool=False,
                    scanner_type:str="FD"
                        ):
        '''
        make all calculators for given model
        return: a dictionary of calculators
        '''
        from calculators import make_calc_all
        self.cal_all = make_calc_all(self,kbT,use_A,scanner_type)
        self.kbT = kbT
        return self.cal_all
    
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
        
    def run_all(self,job_name:str,batch_size:int=100,tabulate:bool=False,save_npz:bool=True,frmsf:bool=False,band_range=None):
        '''
        run all calculators for given k points

        Parameters
        ----------
        job_name : str
            Name prefix for output files.
        batch_size : int
            Number of k-points per batch.
        tabulate : bool
            If True, retain per-k-point results (required for frmsf).
        frmsf : bool
            If True, export results in FermiSurfer .frmsf format.
            Requires tabulate=True and kbT=0 (no Fermi scanning).
        band_range : tuple or None
            (b_min, b_max) 0-based band index range for frmsf output.
            b_min inclusive, b_max exclusive. None means all bands.
        '''
        if self.cal_all is None:
            raise ValueError("cal_all is None, please make calculators first")
        if frmsf and not tabulate:
            raise ValueError("frmsf=True requires tabulate=True")
        if self.kbT != 0.0 and not frmsf:
            raise ValueError("kbT=0 is required for frmsf=True")


        res_dict ={}
        if self.k_points is None:
            raise ValueError("k_points is None, please make k mesh first")
        k_points = self.k_points
        nk = k_points.shape[0]
        V_k =self.V / nk

        from run import run_kps
        Results = run_kps(k_points,self.cal_all,batch_size=batch_size,tabulate=tabulate)
        print(f"Results shape: {Results.shape}")


        res_dict["Energy"] = np.asarray(Results[0])
        
        all_calculators_name = list(self.all_calculators.keys())
        for i,name in enumerate(all_calculators_name):
            if tabulate:
                res_dict[name] = np.asarray(Results[i+1].reshape((self.nkx,self.nky,self.nkz,-1)))
            else:
                res_dict[name] = np.asarray(Results[i+1] * V_k)
        
        import time
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print(f"finish kubo calculation at {time_str}")
        print("------------------------")
        if save_npz:
            np.savez_compressed(f"{job_name}_{time_str}.npz", **res_dict)
            print(f"save to {job_name}_{time_str}.npz")

        # export Fermisurfer .frmsf files
        if frmsf:
            from fermisurfer_io import write_frmsf_all
            native_results = np.asarray(Results)
            write_frmsf_all(self, native_results, job_name, band_range=band_range)
        
        # return
        return res_dict

# test pass
if __name__ == "__main__":
    from TB_io import read_TB_model
    import operators as ops
    model = read_TB_model("TB_Cu.npz")

    # fermi level
    model.EF = 12.43 
    # Or find EF from file
    # model.set_EF("Cu.scf.out")
    model.set_E_fermis(-10,8,0.05)
    # operators
    matrix_all = ops.operator_matrix_all(['spd'])
    Lz = matrix_all.pre_prepared_operators['Lz']
    #calculators
    model.clear_calculator()
    model.make_calc_kubo("OHC_z",alpha=0,beta=1,Gamma=0.025,Operator=Lz)
    model.make_calc_OBC("OHC_z_modern",alpha=0,beta=1,gamma=2,Gamma=0.025)

    #calculators
    #_inner_calculator = model.make_calc_inner()
    #_kubo_calculator =  model.make_calc_kubo()
    model.make_calc_all(kbT=0.003,use_A=True)

    # k_mesh        
    _,_ = model.make_k_mesh(nkx=100,nky=100,nkz=100)
    
    # run 
    model.run_all(job_name="modern",batch_size=1000,tabulate=True,frmsf=True)
