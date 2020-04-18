import numpy as np


def get_neighbors(int natom,list atom_name,float[:,:] rcuta,float[:,:] R):
    cdef int i,j,k,l
    cdef list table = []
    cdef list tableh= []
    cdef list angs  = []
    cdef list tors  = []
    cdef list hbs   = []
    cdef list tab_  = []
    # cdef list tabh_ = []

    for i in range(natom):
        tab_  = []
        tabh_ = []
        for j in range(natom):
            if i!=j:
               if R[i][j]<rcuta[i][j]:
                  tab_.append(j)
               # if R[i][j]<rcut[i][j]:
               #    tabh_.append(j)
        table.append(tab_)
        # tableh.append(tabh_)

    for i in range(natom):
        for j in table[i]:
            for k in table[j]:
                if i!=k:
                   if not [k,j,i] in angs:
                      angs.append([i,j,k])
                   for l in table[k]:
                       if l!=j and l!=i:
                          if not [l,k,j,i] in tors:
                             tors.append([i,j,k,l])
    for i in range(natom):
        for j in table[i]:
            for k in range(natom):
                if atom_name[i]!='H' and atom_name[j]=='H' and atom_name[k]!='H' and k!=i:
                   hbs.append([i,j,k])
    return angs,tors,hbs


def get_pangle(dict p,list atom_name,int np_,list p_ang,int nang,list angs):
    cdef str an,key 
    cdef int i,a_
    cdef list ang
    cdef dict P = {}
    # cdef double[:,:] P = np.zeros([np_,nang]) 
    
    for i in range(np_):
        key = p_ang[i]
        P[key] = np.zeros([nang],dtype=np.float32)
        for a_ in range(nang):
            ang = angs[a_]
            an = key+'_'+atom_name[ang[0]]+'-'+atom_name[ang[1]]+'-'+atom_name[ang[2]]
            if not an in p:
               an = key+'_'+atom_name[ang[2]]+'-'+atom_name[ang[1]]+'-'+atom_name[ang[0]]
            P[key][a_] = p[an]

    P['val3'] = np.zeros([nang],dtype=np.float32)
    for a_ in range(nang):
        ang = angs[a_]
        an = 'val3' + '_' + atom_name[ang[1]] 
        P['val3'][a_] = p[an]

    P['val5'] = np.zeros([nang],dtype=np.float32)
    for a_ in range(nang):
        ang = angs[a_]
        an = 'val5' + '_' + atom_name[ang[1]] 
        P['val5'][a_] = p[an]
    return P


def get_ptorsion(dict p,list atom_name,int np_,list p_tor,int ntor,list tors):
    cdef str tn,key
    cdef int i,t_
    cdef list tor
    cdef dict P = {}
    # cdef double[:,:] P = np.zeros([np_,ntor]) 

    for i in range(np_):
        key = p_tor[i]
        P[key] = np.zeros([ntor],dtype=np.float32)
        for t_ in range(ntor):
            tor = tors[t_]
            tn = key+'_'+atom_name[tor[0]]+'-'+atom_name[tor[1]]+'-'+atom_name[tor[2]]+'-'+atom_name[tor[3]]
            if tn not in p:
               tn = key+'_'+atom_name[tor[3]]+'-'+atom_name[tor[2]]+'-'+atom_name[tor[1]]+'-'+atom_name[tor[0]]
            P[key][t_] = p[tn]
    return P


def get_phb(dict p,list atom_name,int np_,list p_hb,int nhb,list hbs):
    cdef str tn,key
    cdef int i,t_
    cdef list hb
    cdef dict P = {}

    for i in range(np_):
        key = p_hb[i]
        P[key] = np.zeros([nhb],dtype=np.float32)
        for t_ in range(nhb):
            hb = hbs[t_]
            hn = key+'_'+atom_name[hb[0]]+'-'+atom_name[hb[1]]+'-'+atom_name[hb[2]]
            P[key][t_] = p[hn]
    return P
    

