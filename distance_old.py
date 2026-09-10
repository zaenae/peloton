import numpy as np
import gurobipy as gp
from gurobipy import GRB
from generator import (
    make_code, nullspace_mod2, complement_basis,
    row_reduce, reduce_vec )
#def excat
'''import numpy as np
import gurobipy as gp
from generator import make_code, nullspace_mod2, row_reduce, complement_basis'''
def exact_x_distance(Hx, Hz):
    n = Hx.shape[1]
    Hx_basis, _ = row_reduce(Hx)
    Hx_basis = Hx_basis[np.any(Hx_basis, axis=1)]
    ker_hz = nullspace_mod2(Hz)
    Gx = complement_basis(ker_hz, Hx)
    r = Hx_basis.shape[0]
    k = Gx.shape[0]
    model = gp.Model("X_distance")
    a = model.addVars(r, vtype=gp.GRB.BINARY, name="a")
    #d
    q = model.addVars(k,vtype=gp.GRB.BINARY, name="q")
    x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")
    t = model.addVars(n, vtype=gp.GRB.INTEGER, lb=0, name="t")
    for j in range(n):
        stabilizer_part = gp.quicksum(
            int(Hx_basis[i, j]) * a[i] for i in range(r)
        )
        logical_part = gp.quicksum(
            int(Gx[l, j]) * q[l] for l in range(k)
        )
        model.addConstr(
            stabilizer_part + logical_part == x[j] + 2*t[j]
        )
    model.addConstr(gp.quicksum(q[l] for l in range(k)) >= 1)
    model.setObjective(
        gp.quicksum(x[j] for j in range(n)),
        gp.GRB.MINIMIZE
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("X-distance optimization found no solution")
    logical_x = np.array(
        [round(x[j].X) for j in range(n)],
        dtype=np.uint8
    )
    return int(round(model.ObjVal)), logical_x

    """
    
    
    
    exact z solver ffsfsfsfsfsffsfsffs
    
    
    
    
    
    """
def exact_z_distance(Hx, Hz):
    n = Hz.shape[1]
    Hz_basis, _ = row_reduce(Hz)
    Hz_basis = Hz_basis[np.any(Hz_basis, axis=1)]
    ker_hx = nullspace_mod2(Hx)
    Gz = complement_basis(ker_hx, Hz)
    r = Hz_basis.shape[0]
    k = Gz.shape[0]

    model = gp.Model("Z_distance")
    a = model.addVars(r, vtype=gp.GRB.BINARY, name="a")
    q = model.addVars(k, vtype=gp.GRB.BINARY, name="q")
    z = model.addVars(n, vtype=gp.GRB.BINARY, name="z")
    t = model.addVars(n, vtype=gp.GRB.INTEGER, lb=0, name="t")

    for j in range(n):
        stabilizer_part = gp.quicksum(
            int(Hz_basis[i, j]) * a[i] for i in range(r)
        )
        logical_part = gp.quicksum(
            int(Gz[l, j]) * q[l] for l in range(k)
        )
        model.addConstr(
            stabilizer_part + logical_part == z[j] + 2*t[j]
        )

    model.addConstr(gp.quicksum(q[l] for l in range(k)) >= 1)

    model.setObjective(
        gp.quicksum(z[j] for j in range(n)),
        gp.GRB.MINIMIZE
    )

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError("Z-distance optimization found no solution")

    logical_z = np.array(
        [round(z[j].X) for j in range(n)],
        dtype=np.uint8
    )

    return int(round(model.ObjVal)), logical_z    
"""if __name__ == "__main__":
   
    Hx, Hz, ell, m = make_code(1, 1)
    dx, logical_x = exact_x_distance(Hx, Hz)
    print("d_X =", dx)
    print("weight =", int(logical_x.sum()))
if __name__ == "__main__":
    Hx,Hz,ell, m = make_code(1, 1)
    dx,logical_x= exact_x_distance(Hx, Hz)
    print()
    print("exact X distance =", dx)
    print("x logical weight =", int(logical_x.sum()))
    print("x logical =", np.where(logical_x == 1)[0])
"""
if __name__ == "__main__":
    Hx, Hz, ell, m = make_code(1, 1)
    dx, logical_x = exact_x_distance(Hx, Hz)
    print("d_X =", dx)
    print("weight =", int(logical_x.sum()))
    print("Hz x^T =", np.mod(Hz @ logical_x, 2))
    Hx_rref, pivots = row_reduce(Hx)
    reduced = reduce_vec(logical_x, Hx_rref, pivots)
    print("outside X stabilizer =", bool(reduced.any()))
    print("logical support =", np.where(logical_x == 1)[0])

    """ updated test"""
if __name__ == "__main__":
    Hx, Hz, ell, m = make_code(1, 1)

    dz, logical_z = exact_z_distance(Hx, Hz)

    print()
    print("d_Z =", dz)
    print("weight =", int(logical_z.sum()))
    print("Hx z^T =", np.mod(Hx @ logical_z, 2))

    Hz_rref, pivots = row_reduce(Hz)
    reduced = reduce_vec(logical_z, Hz_rref, pivots)
    print("outside Z stabilizer =", bool(reduced.any()))
    print("logical support =", np.where(logical_z == 1)[0])