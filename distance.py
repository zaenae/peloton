"""import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from generator import (
    make_code,
    nullspace_mod2,
    complement_basis,
    row_reduce,
    reduce_vec
)


def build_code(r, b):
    Hx, Hz, ell, m = make_code(r, b)

    return Hx, Hz, ell, m"""

#preprocessing layer for the solver
import sys
import numpy as np

from generator import (
    make_code,
    nullspace_mod2,
    complement_basis,
    row_reduce,
    reduce_vec
)


def clean_rows(A):
    R, pivots = row_reduce(A)
    R = R[np.any(R, axis=1)]
    return R, pivots


def prepare(Hx, Hz):

    # Row spaces / ranks
    sx, px = clean_rows(Hx)
    sz, pz = clean_rows(Hz)

    # Kernels
    ker_z = nullspace_mod2(Hz)
    ker_x = nullspace_mod2(Hx)

    # Logical directions
    lx = complement_basis(ker_z, Hx)
    lz = complement_basis(ker_x, Hz)

    return {
        "n": Hx.shape[1],

        "Hx": Hx,
        "Hz": Hz,

        "sx": sx,
        "sz": sz,

        "ker_x": ker_x,
        "ker_z": ker_z,

        "lx": lx,
        "lz": lz,

        "rank_x": sx.shape[0],
        "rank_z": sz.shape[0],
        "k": Hx.shape[1] - sx.shape[0] - sz.shape[0],

        # Keep these around; the solver may need them.
        "pivots_x": px,
        "pivots_z": pz
    }


def is_x_logical(x, data):

    x = np.asarray(x, dtype=np.uint8)

    if np.any((data["Hz"] @ x) & 1):
        return False

    reduced = reduce_vec(
        x,
        data["sx"],
        data["pivots_x"]
    )

    return bool(np.any(reduced))


def is_z_logical(z, data):

    z = np.asarray(z, dtype=np.uint8)

    if np.any((data["Hx"] @ z) & 1):
        return False

    reduced = reduce_vec(
        z,
        data["sz"],
        data["pivots_z"]
    )

    return bool(np.any(reduced))


def inspect(r, b):

    Hx, Hz, ell, m = make_code(r, b)

    data = prepare(Hx, Hz)

    print()
    print("========== GBB ==========")
    print(f"(r,b) = ({r},{b})")
    print(f"n     = {data['n']}")
    print(f"rankX = {data['rank_x']}")
    print(f"rankZ = {data['rank_z']}")
    print(f"k     = {data['k']}")
    print()

    print("ker(Hz) :", data["ker_z"].shape)
    print("row(Hx) :", data["sx"].shape)
    print("LX      :", data["lx"].shape)
    print()

    print("ker(Hx) :", data["ker_x"].shape)
    print("row(Hz) :", data["sz"].shape)
    print("LZ      :", data["lz"].shape)

    print()
    print(f"ell = {ell}")
    print(f"m   = {m}")
    print("==========================")
    print()



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="GBB quantum LDPC distance solver"
    )

    parser.add_argument("r", type=int)
    parser.add_argument("b", type=int)

    args = parser.parse_args()

    inspect(args.r, args.b)