import numpy as np

FRMSF_VERSION = 1


def _validate_frmsf_inputs(model, results):
    nkx = getattr(model, 'nkx', None)
    nky = getattr(model, 'nky', None)
    nkz = getattr(model, 'nkz', None)
    recip_lattice = getattr(model, 'recip_lattice', None)
    EF = getattr(model, 'EF', None)

    if None in (nkx, nky, nkz):
        raise ValueError("model.nkx/nky/nkz not set. Call model.make_k_mesh() first.")
    if recip_lattice is None:
        raise ValueError("model.recip_lattice is not set.")
    if EF is None:
        raise ValueError("model.EF is not set. Set it manually or use model.set_EF().")
    if EF == 0.0:
        import warnings
        warnings.warn("model.EF is 0.0. Energy values will NOT be shifted relative to the Fermi level.")

    nk = nkx * nky * nkz
    if results.ndim != 3:
        raise ValueError(
            f"results should be 3D (nk, n_operator+1, nwan), got shape {results.shape}"
        )
    if results.shape[0] != nk:
        raise ValueError(
            f"results first dim {results.shape[0]} != nk {nk}"
        )


def _write_header(f, nkx, nky, nkz, nband, recip_lattice):
    f.write(f"{nkx} {nky} {nkz}\n")
    f.write(f"{FRMSF_VERSION}\n")
    f.write(f"{nband}\n")
    for i in range(3):
        r = recip_lattice[i]
        f.write(f"{r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n")


def _write_data_block(f, data_4d, nkx, nky, nkz, nband):
    """
    Write data in frmsf order: z fastest, then y, then x, then band.

    data_4d: ndarray of shape (nkx, nky, nkz, nband)
    """
    for iband in range(nband):
        for ix in range(nkx):
            for iy in range(nky):
                for iz in range(nkz):
                    f.write(f"{data_4d[ix, iy, iz, iband]:15.8f}\n")


def write_frmsf_energy(model, results, filepath):
    """
    Write a .frmsf file containing only band energies (E - EF).

    Parameters
    ----------
    model : TB_model
        The tight-binding model instance (must have nkx, nky, nkz, recip_lattice, EF).
    results : np.ndarray
        Tabulated results, shape (nk, 1+n_calc, nwan).
        results[:, 0, :] must contain band energies.
    filepath : str
        Output file path.
    """
    _validate_frmsf_inputs(model, results)

    nkx, nky, nkz = model.nkx, model.nky, model.nkz
    nwan = model.nwan
    EF = model.EF

    energy_flat = np.asarray(results[:, 0, :], dtype=np.float64)
    energy_4d = energy_flat.reshape(nkx, nky, nkz, nwan)
    energy_4d = energy_4d - EF

    with open(filepath, 'w') as f:
        _write_header(f, nkx, nky, nkz, nwan, np.asarray(model.recip_lattice))
        _write_data_block(f, energy_4d, nkx, nky, nkz, nwan)

    file_size_mb = nkx * nky * nkz * nwan * 18 / (1024 * 1024)
    print(f"Energy .frmsf written to {filepath} (~{file_size_mb:.1f} MB)")


def write_frmsf_quantity(model, results, calc_index, calc_name, filepath):
    """
    Write a .frmsf file containing band energies (E - EF) and one physical
    quantity overlay.

    Parameters
    ----------
    model : TB_model
        The tight-binding model instance.
    results : np.ndarray
        Tabulated results, shape (nk, 1+n_calc, nwan).
    calc_index : int
        Index of the calculator in results (0-based).
        results[:, 1+calc_index, :] contains the physical quantity.
    calc_name : str
        Name of the physical quantity (used in log message).
    filepath : str
        Output file path.
    """
    _validate_frmsf_inputs(model, results)

    nkx, nky, nkz = model.nkx, model.nky, model.nkz
    nwan = model.nwan
    EF = model.EF

    energy_flat = np.asarray(results[:, 0, :], dtype=np.float64)
    energy_4d = energy_flat.reshape(nkx, nky, nkz, nwan)
    energy_4d = energy_4d - EF

    quantity_flat = np.asarray(results[:, 1 + calc_index, :], dtype=np.float64)
    quantity_4d = quantity_flat.reshape(nkx, nky, nkz, nwan)

    with open(filepath, 'w') as f:
        _write_header(f, nkx, nky, nkz, nwan, np.asarray(model.recip_lattice))
        _write_data_block(f, energy_4d, nkx, nky, nkz, nwan)
        _write_data_block(f, quantity_4d, nkx, nky, nkz, nwan)

    file_size_mb = nkx * nky * nkz * nwan * 2 * 18 / (1024 * 1024)
    print(f"{calc_name} .frmsf written to {filepath} (~{file_size_mb:.1f} MB)")


def write_frmsf_all(model, results, output_prefix):
    """
    Write .frmsf files for all registered calculators.

    Produces one file per calculator, each containing energy data (E - EF)
    followed by the physical quantity overlay.

    Parameters
    ----------
    model : TB_model
        The tight-binding model instance.
        Must have model.all_calculators populated.
    results : np.ndarray
        Tabulated results, shape (nk, 1+n_calc, nwan).
    output_prefix : str
        Prefix for output files, e.g. "Cu" produces "Cu_OHC_z.frmsf".
    """
    _validate_frmsf_inputs(model, results)

    all_calculators_name = list(model.all_calculators.keys())
    n_calc = len(all_calculators_name)

    if results.shape[1] != n_calc + 1:
        raise ValueError(
            f"results shape[1]={results.shape[1]} "
            f"does not match n_calc+1={n_calc + 1}"
        )

    energy_path = f"{output_prefix}_Energy.frmsf"
    write_frmsf_energy(model, results, energy_path)

    for i, name in enumerate(all_calculators_name):
        quantity_path = f"{output_prefix}_{name}.frmsf"
        write_frmsf_quantity(model, results, i, name, quantity_path)
