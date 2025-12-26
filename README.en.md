# Tight-Binding Model Toolkit (TB_model)

## Project Overview
This is a toolkit for tight-binding model calculations, providing multiple computational functions including Hamiltonian calculation, Kubo formula computation, density of states (DOS) calculation, orbital magnetization calculation, orbital Berry curvature calculation, and more. The project supports GPU acceleration to efficiently handle complex physical computations.

## Main Features
- Construct k-space paths and grids
- Compute Hamiltonian and its velocity
- Implement Kubo formula calculations
- Support for density of states (DOS) and projected density of states (PDOS) calculations
- Provide band structure calculations along high-symmetry paths
- Calculate orbital magnetization (modern theory)
- Calculate orbital Berry curvature (OBC)
- Support GPU-accelerated computations
- Support gauge potential (AA_R) terms

## Project Structure
```
.
├── K_space.py          # k-space path and grid construction
├── TB_io.py            # Model data I/O
├── TB_model.py         # Tight-binding model data structure
├── calculators.py      # Calculator generation module
├── Hk.py               # Hamiltonian calculation module
├── documentation.md    # Project documentation
├── operators.py        # Operator construction module
├── run.py              # Computation execution module
├── get_WB.py           # WannierBerri data reading tool
├── utility.py          # Utility functions
└── job.py              # Example run script
```

## Core Module Descriptions

### TB_model.py
Defines the `TB_model` data class for storing fundamental tight-binding model information, including orbital details, Hamiltonian parameters, gauge potential, and more.

### calculators.py
Provides multiple calculator generation functions:
- `make_k_hamiltonian_calculator`: Create a k-space Hamiltonian calculator
- `make_k_hamiltonian_velocity_calculator`: Create a Hamiltonian velocity calculator
- `make_calc_inner`: Create an inner-product calculator (for DOS computation)
- `make_calc_kubo`: Create a Kubo formula calculator
- `make_calc_OBC`: Create an orbital Berry curvature calculator
- `make_calc_mag_modern`: Calculate orbital magnetization using modern theory
- `make_calc_mag_matrix`: Calculate orbital magnetization matrix
- `make_calc_all`: Unified interface integrating all computational functions
- `make_calc_mag_scan`: Magnetization scan calculator

### Hk.py
Hamiltonian calculation module:
- `make_k_hamiltonian_calculator`: Create a k-space Hamiltonian calculator
- `make_calc_AA_k`: Calculate k-space gauge potential AA_k
- `make_k_hamiltonian_velocity_calculator`: Create a Hamiltonian velocity calculator

### K_space.py
Contains functions for constructing k-space paths and grids:
- `build_k_path`: Construct a k-path based on given high-symmetry points
- `build_k_mesh`: Construct a 3D k-grid

### operators.py
Provides operator construction capabilities:
- `operator_matrix_all`: Operator matrix class
- `build_operator_all`: Build complete operator matrices
- `build_operator_LS`: Build angular momentum operators
- `build_P_easy`: Build projection operators

### utility.py
Includes utility functions:
- `fermi_dirac`: Fermi-Dirac distribution function
- `fermi_dirac_derivative`: Derivative of Fermi-Dirac distribution
- `cal_center_matrix`: Compute center matrix
- `find_EF`: Locate the Fermi level

### TB_io.py
Implements model data reading and writing:
- `read_TB_model`: Read model data from file
- `save_TB_model`: Save model data to file

### run.py
Provides interface functions for running computations:
- `run_kps`: Execute k-space point computations, supporting batch processing and GPU acceleration

### get_WB.py
WannierBerri data reading tool:
- Read data from WannierBerri's _tb.dat file
- Extract HR, cRvec, real_lattice, recip_lattice, Wan_center, AA_R and other data
- Save in npz format for TB_model use

## Usage Guide

### Basic Workflow
1. Load the model: `model = read_TB_model("TB.npz")`
2. Set Fermi level: `model.set_EF("Cu.scf.out")` or directly set `model.EF = 12.43`
3. Clear previous calculators (optional): `model.clear_calculator()`
4. Add operators and calculators
5. Create unified calculator: `model.make_calc_all(kbT=0.026, use_A=False)`
6. Build k-grid: `model.make_k_mesh(100, 100, 100)`
7. Run computation: `results = model.run_all("job_name", batch_size=1000, tabulate=False)`

### Add Inner-Product Operator (for DOS Calculation)
```python
from utility import build_P_easy
from calculators import make_calc_inner

P = build_P_easy(model.orbital_list, "all")
model.make_calc_inner("cumDOS", jnp.eye(model.nwan))
```

### Add Operators for Kubo Calculation
```python
import operators as ops

matrix_all = ops.operator_matrix_all(['spd'])
Lz = matrix_all.pre_prepared_operators["Lz"]
model.make_calc_kubo("OHC_z", alpha=0, beta=1, Gamma=0.025, Operator=Lz)
```

### Create Orbital Berry Curvature Calculator
```python
model.make_calc_OBC("OBC_z", alpha=0, beta=1, gamma=2, Gamma=0.025)
```

### Create Unified Calculator
```python
model.make_calc_all(
    kbT=0.026,        # Temperature in eV
    use_A=False       # Whether to include gauge potential terms in velocity operator
)
```

### Run Kubo Calculation
```python
from K_space import build_k_mesh
from run import run_kps

k_mesh = build_k_mesh(model, 10, 10, 1)
results = run_kps(k_mesh, calc_kubo, batch_size=100)
```

### Or Use the Model's run_all Method to Run All Calculations
```python
results = model.run_all("job_name", calculators)
```

## Advanced Features

### Projected Density of States (PDOS) Calculation
#### Create an s-orbital Projection Operator
```python
from utility import build_P_easy

P_s = matrix_all.make_P_easy("s")
model.make_calc_inner("s_PDOS", P_s)
```

### Band Structure Along High-Symmetry Paths
#### Define High-Symmetry Points (Example: Cubic Crystal)
```python
from K_space import build_k_path

k_points = jnp.array([
    [0.0, 0.0, 0.0],   # Γ point
    [0.5, 0.0, 0.0],   # X point
    [0.5, 0.5, 0.0],   # M point
    [0.0, 0.0, 0.0],   # Γ point
])
ticks_label = ["Γ", "X", "M", "Γ"]
```

#### Construct k-Path
```python
k_path = build_k_path(model, k_points, ticks_label, length=100)
```

### Orbital Magnetization Calculation
Calculate orbital magnetization using modern theory:
```python
model.make_calc_OBC("OBC_z", alpha=0, beta=1, gamma=2, Gamma=0.025)

# Or use magnetization scan calculator
mag_scan_calculator = model.make_calc_mag_scan(kbT=0.003, use_A=False, gamma=2)
model.add_calculator(mag_scan_calculator, "mag_scan")
```

### Use Gauge Potential Terms
Include gauge potential terms in velocity operator:
```python
model.make_calc_all(kbT=0.026, use_A=True)
```

### GPU Acceleration
The `run_kps` function automatically detects GPU availability. For optimal performance, set `batch_size` to be divisible by the number of GPUs.

## Technical Features
- Uses JAX for automatic differentiation and GPU acceleration
- Supports large-scale parallel computation
- Modular design for easy extensibility
- Includes detailed documentation and examples
- Supports modern orbital magnetization theory and orbital Berry curvature calculations
- Supports gauge potential terms

## Dependencies
- Python 3.8+
- JAX
- NumPy
- dataclasses
- tqdm (progress display)
- Matplotlib (optional, for visualization)
- WannierBerri (optional, for data conversion)

## Performance Optimization Tips
1. Prefer GPU-based computation whenever possible
2. Set `batch_size` appropriately to fully utilize hardware resources
3. Use caching mechanisms for repeated computations
4. Simplify orbital information during model construction to remove unnecessary data
5. Using JIT-compiled functions can significantly improve computational speed

## Example Applications
Refer to the example code in `documentation.md`.

## Notes
1. Ensure input file formats are correct
2. Be mindful of memory limits when using GPU acceleration
3. Adjust precision settings appropriately for complex calculations
4. The use of gauge potential terms (AA_R) will affect the calculation results of velocity operators
5. For complex calculations, it is recommended to test on a small-scale k-grid first

## Extension and Development
The project is designed with a highly modular structure, allowing easy addition of new computational functions and physical models. We recommend following the existing code style for extensions and development.

## License
This project is licensed under the MIT License.

## Contact
Refer to the contact information in `documentation.md`.

## References
1. Pezo et al., PHYSICAL REVIEW B 106, 104414 (2022) - Modern orbital magnetization theory
2. WannierBerri - Wannier interpolation and Berry phase calculation toolkit