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

#gets the nullspace basis mod 2 needed to find logical ops
def nullspace_mod2(mat):
    mat=mat.copy()%2
    rows,cols=mat.shape
    m=mat.copy()
    pivot_cols=[]
    r=0
    for c in range(cols):
        piv=None
        for rr in range(r,rows):
            if m[rr,c]:
                piv=rr
                break
        if piv is None:
            continue
        m[[r,piv]]=m[[piv,r]]
        for rr in range(rows):
            if rr!=r and m[rr,c]:
                m[rr]=(m[rr]+m[r])%2
        pivot_cols.append(c)
        r+=1
    free_cols=[c for c in range(cols) if c not in pivot_cols]
    basis=[]
    for fc in free_cols:
        vec=np.zeros(cols,dtype=np.uint8)
        vec[fc]=1
        for i,pc in enumerate(pivot_cols):
            vec[pc]=m[i,fc]
        basis.append(vec)
    return basis

def row_reduce(mat):
    mat=mat.copy()%2
    rows,cols=mat.shape
    r=0
    pivot_cols=[]
    for c in range(cols):
        piv=None
        for rr in range(r,rows):
            if mat[rr,c]:
                piv=rr
                break
        if piv is None:
            continue
        mat[[r,piv]]=mat[[piv,r]]
        for rr in range(rows):
            if rr!=r and mat[rr,c]:
                mat[rr]=(mat[rr]+mat[r])%2
        pivot_cols.append(c)
        r+=1
    return mat[:r],pivot_cols

def reduce_vec(vec,rref,pivot_cols):
    vec=vec.copy()%2
    for i,pc in enumerate(pivot_cols):
        if vec[pc]:
            vec=(vec+rref[i])%2
    return vec

#finds k logical ops that commute w the other check type but arent stabilizeRs
def find_logicals(h_this_type,h_other_type,k):
    ker=nullspace_mod2(h_other_type)
    rref,pivots=row_reduce(h_this_type)
    logicals=[]
    for v in ker:
        red=reduce_vec(v,rref,pivots)
        if red.any():
            logicals.append(v)
        if len(logicals)==k:
            break
    return logicals


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
    #qquick test on the two codes
    test_cases=[(1,1),(2,0)] #gross& two-gross

for r,b in test_cases:
    hx,hz,ell,m =make_code(r,b)
    n=hx.shape[1]

    check= (hx @ hz.T)%2
    if check.any():
        print("hx and hz dont commute, WRONG")
        continue

    #every check should touch 6 qubits - 3 from Aand 3 from B if not something's off
    rw =hx.sum(axis=1)
    rwz =hz.sum(axis=1)
    if not (rw==6).all() or not (rwz==6).all():
        print("check the construction")
        continue

    k = n-rank_mod2(hx)-rank_mod2(hz)
    print("r =",r,"b =",b," ell=",ell,"m=",m,"  n=",n,"k=",k)

    x_logicals=find_logicals(hx,hz,k)
    z_logicals=find_logicals(hz,hx,k)
    print("  found",len(x_logicals),"x logicals and",len(z_logicals),"z logicals")

    save_code(hx,hz,r,b)