import numpy as np
import argparse
import os

#values from the  tour de gross paper ssssssssssssssssss
#appendix A.1A = 1 + y + x^3 y^-1 , B = 1 + x + x^-1 y^-3

A_terms =[(0,0),(0,1),(3,-1)]
B_terms =[(0,0),(1,0),(-1,-3)]

def make_code(r,b):
    ell =6*(r+b)
    m = 6*r
    ncells= ell*m

    #l qubits are 0 to ncells-1 andr qubits are ncells to 2*ncells-1
    def cell(i,j):
        return (i%ell)*m + (j%m)

    hx=np.zeros((ncells,2*ncells),dtype=np.uint8)
    hz=np.zeros((ncells,2*ncells),dtype=np.uint8)

    for i in range(ell):
        for j in range(m):
            c= cell(i,j)
            for p,q in A_terms:
                hx[c,cell(i+p,j+q)] = 1 
            for p,q in B_terms:
                hx[c,ncells+cell(i+p,j+q)] = 1 #Rqubit

            #z checks connect via the transposed polys
                hz[c,cell(i-p,j-q)] = 1
            for p,q in A_terms:
                hz[c,ncells+cell(i-p,j-q)] = 1

    return hx,hz,ell,m


def rank_mod2(mat):
    mat =mat.copy()%2
    rows,cols= mat.shape
    rank=0
    for col in range(cols):
        piv=-1
        for row in range(rank,rows):
            if mat[row][col]==1:
                piv=row
                break
        if piv==-1:
            continue
        mat[[rank,piv]]=mat[[piv,rank]]
        for row in range(rows):
            if row!=rank and mat[row][col]==1:
                mat[row]=(mat[row]+mat[rank])%2
        rank+=1
    return rank


def save_code(hx,hz,r,b):
    #dumpsmatrices for gurobi load
    os.makedirs("codes",exist_ok=True)
    np.save("codes/hx_r"+str(r)+"_b"+str(b)+".npy",hx)
    np.save("codes/hz_r"+str(r)+"_b"+str(b)+".npy",hz)


parser=argparse.ArgumentParser()
parser.add_argument("--r",type=int,default=None)
parser.add_argument("--b",type=int,default=None)
args=parser.parse_args()

if args.r is not None and args.b is not None:
    test_cases=[(args.r,args.b)]
else:
    # quick test on the two codes we actually know the answer for
    test_cases=[(1,1),(2,0)] #gross& two-gross

for r,b in test_cases:
    hx,hz,ell,m =make_code(r,b)
    n=hx.shape[1]

    check= (hx @ hz.T)%2
    if check.any():
        print(" hx and hz dont commute, something is wrong")
        continue

    #every check should touch 6 qubits (3 from A, 3 from B), if not something's off
    rw =hx.sum(axis=1)
    rwz =hz.sum(axis=1)
    if not (rw==6).all() or not (rwz==6).all():
        print("check the construction")
        continue

    k = n-rank_mod2(hx)-rank_mod2(hz)
    print("r =",r,"b =",b," ell=",ell,"m=",m,"  n=",n,"k=",k)

    save_code(hx,hz,r,b)
