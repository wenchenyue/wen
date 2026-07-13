# Parse Wannier90 *_tb.dat files and save extracted matrices to .npz (pure numpy, no wannierberri)
import numpy as np
import os
import argparse


def _compute_recip_lattice(real_lattice):
    """Reciprocal lattice: B = 2π * (A^{-1})^T"""
    return 2.0 * np.pi * np.linalg.inv(real_lattice).T


def parse_tb_file(tb_file, convention_II_to_I=True, include_AA=True):
    """
    Parse a Wannier90 *_tb.dat file and return a dict of extracted matrices.

    param tb_file: path to *_tb.dat file
    param convention_II_to_I: if True, convert AA_R from Wannier90 convention II to convention I
    param include_AA: whether to parse and return AA_R
    return: dict with keys HR, iRvec, cRvec, real_lattice, recip_lattice, Wan_center, num_wann, [AA_R]
    """
    if not os.path.exists(tb_file):
        raise FileNotFoundError(f"File not found: {tb_file}")

    print(f"[parse_tb_file] Parsing {tb_file} ...")

    f = open(tb_file, "r")

    # header line
    header = f.readline().strip()
    print(f"  header: {header}")

    # 3 lattice vectors (Angstrom)
    real_lattice = np.array(
        [f.readline().split()[:3] for _ in range(3)],
        dtype=float
    )
    print(f"  real_lattice:\n{real_lattice}")

    num_wann = int(f.readline())
    nRvec = int(f.readline())
    print(f"  num_wann = {num_wann}, nRvec = {nRvec}")

    # degeneracy weights, may span multiple lines
    Ndegen = []
    while len(Ndegen) < nRvec:
        Ndegen += f.readline().split()
    Ndegen = np.array(Ndegen, dtype=int)

    # parse Ham_R block — Fortran column-major order, transpose to C row-major
    iRvec = []
    Ham_R = np.zeros((nRvec, num_wann, num_wann), dtype=complex)
    for ir in range(nRvec):
        f.readline()  # blank separator
        iRvec.append(f.readline().split())
        hh = np.array(
            [[f.readline().split()[2:4] for _ in range(num_wann)]
             for _ in range(num_wann)],
            dtype=float
        ).transpose((1, 0, 2))
        Ham_R[ir] = (hh[:, :, 0] + 1j * hh[:, :, 1]) / Ndegen[ir]

    iRvec = np.array(iRvec, dtype=int)
    print(f"  Ham_R.shape = {Ham_R.shape}, iRvec.shape = {iRvec.shape}")

    # index of the home cell R = [0, 0, 0]
    iR0 = iRvec.tolist().index([0, 0, 0])

    # parse AA_R block (position matrix elements)
    AA_R = None
    wannier_centers_cart = None

    if include_AA:
        AA_R = np.zeros((nRvec, num_wann, num_wann, 3), dtype=complex)
        for ir in range(nRvec):
            f.readline()
            assert np.all(np.array(f.readline().split(), dtype=int) == iRvec[ir]), \
                f"R vector mismatch at ir={ir}"
            # columns 2-7: x_re, x_im, y_re, y_im, z_re, z_im
            aa = np.array(
                [[f.readline().split()[2:8] for _ in range(num_wann)]
                 for _ in range(num_wann)],
                dtype=float
            )
            AA_R[ir] = (aa[:, :, 0::2] + 1j * aa[:, :, 1::2]).transpose((1, 0, 2)) / Ndegen[ir]

        # Wannier centers from diagonal of AA_R at R=0
        wannier_centers_cart = np.diagonal(AA_R[iR0], axis1=0, axis2=1).T.copy().real
        print(f"  AA_R.shape = {AA_R.shape}")
        print(f"  Wan_center:\n{wannier_centers_cart}")

        if convention_II_to_I:
            # subtract on-site centers so AA_R[iR0] diagonal becomes zero (convention I)
            AA_R[iR0, np.arange(num_wann), np.arange(num_wann), :] -= wannier_centers_cart

    else:
        # skip AA_R blocks up to iR0, then read only the R=0 diagonal for Wannier centers
        for ir in range(iR0):
            f.readline()
            assert np.all(np.array(f.readline().split(), dtype=int) == iRvec[ir])
            for _ in range(num_wann ** 2):
                f.readline()

        f.readline()
        assert np.all(np.array(f.readline().split(), dtype=int) == iRvec[iR0])
        aa = np.array(
            [[f.readline().split()[2:8] for _ in range(num_wann)]
             for _ in range(num_wann)],
            dtype=float
        )
        aa = (aa[:, :, 0::2] + 1j * aa[:, :, 1::2]).transpose((1, 0, 2)) / Ndegen[iR0]
        wannier_centers_cart = np.diagonal(aa, axis1=0, axis2=1).T.real
        print(f"  Wan_center:\n{wannier_centers_cart}")

    f.close()

    # Cartesian R vectors and reciprocal lattice
    cRvec = iRvec.dot(real_lattice)
    recip_lattice = _compute_recip_lattice(real_lattice)

    result = {
        'HR': Ham_R,
        'iRvec': iRvec,
        'cRvec': cRvec,
        'real_lattice': real_lattice,
        'recip_lattice': recip_lattice,
        'Wan_center': wannier_centers_cart,
        'num_wann': num_wann,
    }
    if AA_R is not None:
        result['AA_R'] = AA_R

    return result


def save_to_npz(data, output):
    """
    Save the dict returned by parse_tb_file to a .npz file.

    param data: dict from parse_tb_file
    param output: output .npz path
    return: output path
    """
    np.savez(output, **data)
    print(f"[save_to_npz] Saved to {output}")
    _print_summary(data)
    return output


def load_from_npz(npz_path):
    """
    Load a .npz file produced by save_to_npz.

    param npz_path: path to .npz file
    return: dict with the stored arrays
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    print(f"[load_from_npz] Loading {npz_path} ...")
    data = dict(np.load(npz_path, allow_pickle=False))

    required = ['HR', 'iRvec', 'cRvec', 'real_lattice', 'recip_lattice', 'Wan_center', 'num_wann']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    data['num_wann'] = int(data['num_wann'])
    print("[load_from_npz] Done.")
    _print_summary(data)
    return data


def _print_summary(data):
    print("-" * 50)
    print(f"  num_wann       = {data['num_wann']}")
    print(f"  real_lattice   =\n{np.round(data['real_lattice'], 6)}")
    print(f"  recip_lattice  =\n{np.round(data['recip_lattice'], 6)}")
    print(f"  Wan_center     =\n{np.round(data['Wan_center'], 6)}")
    print(f"  HR.shape       = {data['HR'].shape}")
    print(f"  iRvec.shape    = {data['iRvec'].shape}")
    print(f"  iRvec (first 5):\n{data['iRvec'][:5]}")
    print(f"  cRvec.shape    = {data['cRvec'].shape}")
    if 'AA_R' in data:
        print(f"  AA_R.shape     = {data['AA_R'].shape}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Extract matrices from a Wannier90 *_tb.dat file and save as .npz (pure numpy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tb_parser.py wannier90_tb.dat
  python tb_parser.py wannier90_tb.dat -o my_system.npz
  python tb_parser.py wannier90_tb.dat --no-aa
        """
    )
    parser.add_argument("tb_file", help="path to *_tb.dat file")
    parser.add_argument("-o", "--output", default=None,
                        help="output .npz path (default: same directory, same stem)")
    parser.add_argument("--no-aa", action="store_true",
                        help="skip AA_R (position matrix) parsing")
    parser.add_argument("--no-conv", action="store_true",
                        help="skip convention II -> I conversion for AA_R")

    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.tb_file)[0]
        args.output = base + ".npz"

    data = parse_tb_file(
        tb_file=args.tb_file,
        convention_II_to_I=not args.no_conv,
        include_AA=not args.no_aa,
    )
    save_to_npz(data, args.output)


if __name__ == "__main__":
    main()
