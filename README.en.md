# TB_model - Tight-Binding Model Toolkit

## Overview

TB_model is a Python toolkit for tight-binding model calculations, providing various physical quantity computations with GPU acceleration support.

## Main Features

- Band structure calculations
- Density of states (DOS) and projected density of states (PDOS)
- Kubo formula transport calculations
- Orbital magnetization calculations
- Orbital Berry curvature (OBC) calculations
- k-space path and mesh construction

## Quick Start

### Install Dependencies

```bash
pip install jax numpy tqdm
```

### Basic Usage

```python
from TB_io import read_TB_model
from K_space import build_k_mesh

# Load model
model = read_TB_model("TB.npz")

# Set Fermi level
model.EF = 12.43

# Create k-mesh
k_mesh = build_k_mesh(model, 100, 100, 100)

# Run calculations
results = model.run_all("job_name", batch_size=1000)
```

## Documentation

For detailed documentation, see [documentation.md](documentation.md)

## License

MIT License
