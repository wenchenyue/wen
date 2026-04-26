import numpy as np

FRMSF_VERSION = 1


def _apply_band_range(results, nwan, band_range):
    if band_range is None:
        return results, nwan
    b_min, b_max = band_range
    if b_min < 0 or b_max > nwan or b_min >= b_max:
        raise ValueError(
            f"band_range ({b_min}, {b_max}) invalid for nwan={nwan}. "
            f"Expected 0 <= b_min < b_max <= {nwan}."
        )
    return results[..., b_min:b_max], b_max - b_min


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
            f"results should be 3D (n_operator+1, nk, nwan), got shape {results.shape}"
        )
    if results.shape[1] != nk:
        raise ValueError(
            f"results.shape[1]={results.shape[1]} != nk {nk}"
        )


def _write_header(f, nkx, nky, nkz, nband, recip_lattice):
    f.write(f"{nkx} {nky} {nkz}\n")
    f.write(f"{FRMSF_VERSION}\n")
    f.write(f"{nband}\n")
    for i in range(3):
        r = recip_lattice[i]
        f.write(f"{r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n")


def _write_data_block(f, data_4d):
    flat = np.asarray(data_4d).transpose(3, 0, 1, 2).ravel()
    np.savetxt(f, flat, fmt='%15.8f')


def write_frmsf_energy(model, results, filepath, band_range=None):
    _validate_frmsf_inputs(model, results)

    nkx, nky, nkz = model.nkx, model.nky, model.nkz
    nwan = model.nwan
    EF = model.EF

    results, nwan_out = _apply_band_range(results, nwan, band_range)

    energy_flat = np.asarray(results[0], dtype=np.float64)
    energy_4d = energy_flat.reshape(nkx, nky, nkz, nwan_out)
    energy_4d = energy_4d - EF

    with open(filepath, 'w') as f:
        _write_header(f, nkx, nky, nkz, nwan_out, np.asarray(model.recip_lattice))
        _write_data_block(f, energy_4d)

    file_size_mb = nkx * nky * nkz * nwan_out * 18 / (1024 * 1024)
    print(f"Energy .frmsf written to {filepath} (~{file_size_mb:.1f} MB)")


def write_frmsf_quantity(model, results, calc_index, calc_name, filepath, band_range=None):
    _validate_frmsf_inputs(model, results)

    nkx, nky, nkz = model.nkx, model.nky, model.nkz
    nwan = model.nwan
    EF = model.EF

    results, nwan_out = _apply_band_range(results, nwan, band_range)

    energy_flat = np.asarray(results[0], dtype=np.float64)
    energy_4d = energy_flat.reshape(nkx, nky, nkz, nwan_out)
    energy_4d = energy_4d - EF

    quantity_flat = np.asarray(results[1 + calc_index], dtype=np.float64)
    quantity_4d = quantity_flat.reshape(nkx, nky, nkz, nwan_out)

    with open(filepath, 'w') as f:
        _write_header(f, nkx, nky, nkz, nwan_out, np.asarray(model.recip_lattice))
        _write_data_block(f, energy_4d)
        _write_data_block(f, quantity_4d)

    file_size_mb = nkx * nky * nkz * nwan_out * 2 * 18 / (1024 * 1024)
    print(f"{calc_name} .frmsf written to {filepath} (~{file_size_mb:.1f} MB)")


def write_frmsf_all(model, results, output_prefix, band_range=None):
    _validate_frmsf_inputs(model, results)

    all_calculators_name = list(model.all_calculators.keys())
    n_calc = len(all_calculators_name)

    if results.shape[0] != n_calc + 1:
        raise ValueError(
            f"results.shape[0]={results.shape[0]} "
            f"does not match n_calc+1={n_calc + 1}"
        )

    energy_path = f"{output_prefix}_Energy.frmsf"
    write_frmsf_energy(model, results, energy_path, band_range=band_range)

    for i, name in enumerate(all_calculators_name):
        quantity_path = f"{output_prefix}_{name}.frmsf"
        write_frmsf_quantity(model, results, i, name, quantity_path, band_range=band_range)
