# Tight-Binding Model Toolkit (TB_model)

## Project Overview
This is a toolkit for tight-binding model calculations, providing multiple computational functions including Hamiltonian calculation, Kubo formula computation, density of states (DOS) calculation, and more. The project supports GPU acceleration to efficiently handle complex physical computations.

## Main Features
- Construct k-space paths and grids
- Compute Hamiltonian and its velocity
- Implement Kubo formula calculations
- Support for density of states (DOS) and projected density of states (PDOS) calculations
- Provide band structure calculations along high-symmetry paths
- Support GPU-accelerated computations

## Project Structure
```
.
├── K_space.py          # k-space path and grid construction
├── TB_io.py            # Model data I/O
├── TB_model.py         # Tight-binding model data structure
├── calculators.py      # Calculator generation module
├── documentation.md    # Project documentation
├── operators.py        # Operator construction module
├── run.py              # Computation execution module
├── utility.py          # Utility functions
```

## Core Module Descriptions

### TB_model.py
Defines the `TB_model` data class for storing fundamental tight-binding model information, including orbital details and Hamiltonian parameters.

### calculators.py
Provides multiple calculator generation functions:
- `make_k_hamiltonian_calculator`: Create a k-space Hamiltonian calculator
- `make_k_hamiltonian_velocity_calculator`: Create a Hamiltonian velocity calculator
- `make_calc_inner`: Create an inner-product calculator (for DOS computation)
- `make_calc_kubo`: Create a Kubo formula calculator

### K_space.py
Contains functions for constructing k-space paths and grids:
- `build_k_path`: Construct a k-path based on given high-symmetry points
- `build_k_mesh`: Construct a 3D k-grid

### operators.py
Provides operator construction capabilities:
- `operator_matrix_all`: Operator matrix class
- `build_operator_all`: Build complete operator matrices
- `build_operator_LS`: Build angular momentum operators

### utility.py
Includes utility functions:
- `fermi_dirac`: Fermi-Dirac distribution function
- `cal_center_matrix`: Compute center matrix
- `find_EF`: Locate the Fermi level

### TB_io.py
Implements model data reading and writing:
- `read_TB_model`: Read model data from file
- `save_TB_model`: Save model data to file

### run.py
Provides interface functions for running computations:
- `run_kps`: Execute k-space point computations, supporting batch processing and GPU acceleration

## Usage Guide

### Basic Workflow
1. Load the model: `model = read_TB_model("TB.npz")`
2. Create a calculator: `calculator = make_calc_kubo(model)`
3. Construct a k-path: `kps = build_k_path(model, k_points, ticks_label)`
4. Run computation: `results = run_kps(kps, calculator)`

### Add Inner-Product Operator (for DOS Calculation)
```python
from utility import build_P_easy
from calculators import make_calc_inner

P = build_P_easy(model.orbital_list, "all")
calc_inner = make_calc_inner(model, P)
```

### Add Operators for Kubo Calculation
```python
from calculators import make_calc_kubo

calc_kubo = make_calc_kubo(model, alpha=0, beta=1, Gamma=0.025)
```

### Create Inner-Product Calculator
```python
calc_inner = make_calc_inner(model, kbT=0.01)
```

### Create Kubo Calculator
```python
calc_kubo = make_calc_kubo(model, alpha=0, beta=1, Gamma=0.025, kbT=0.01)
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
results = model.run_all(kps, calculators)
```

## Advanced Features

### Projected Density of States (PDOS) Calculation
#### Create an s-orbital Projection Operator
```python
from utility import build_P_easy

P_s = build_P_easy(model.orbital_list, "s")
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

### GPU Acceleration
The `run_kps` function automatically detects GPU availability. For optimal performance, set `batch_size` to be divisible by the number of GPUs.

## Technical Features
- Uses JAX for automatic differentiation and GPU acceleration
- Supports large-scale parallel computation
- Modular design for easy extensibility
- Includes detailed documentation and examples

## Dependencies
- Python 3.8+
- JAX
- NumPy
- dataclasses

## Performance Optimization Tips
1. Prefer GPU-based computation whenever possible
2. Set `batch_size` appropriately to fully utilize hardware resources
3. Use caching mechanisms for repeated computations
4. Simplify orbital information during model construction to remove unnecessary data

## Example Applications
Refer to the example code in `documentation.md`.

## Notes
1. Ensure input file formats are correct
2. Be mindful of memory limits when using GPU acceleration
3. Adjust precision settings appropriately for complex calculations

## Extension and Development
The project is designed with a highly modular structure, allowing easy addition of new computational functions and physical models. We recommend following the existing code style for extensions and development.

## License
This project is licensed under the MIT License.

## Contact
Refer to the contact information in `documentation.md`.