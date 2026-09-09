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