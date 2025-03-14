import numpy as np

class Stack():
    def __init__(self, entry=[]):
        self.entry = entry

    def push(self, x):
        self.entry.append(x)

    def pop(self):
        return self.entry.pop()

    def close(self):
        self.entry = None

def load_density_energy(findi='Individuals'):
    gene = {}
    density = []
    id_g = {}
    dens_g = {}
    op_g = {}
    energy_g= {}

    with open(findi) as f:
        for line in f.readlines():
            st = Stack([])
            for x in line:
                if x != ']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ != '[':
                        x_ = st.pop()
            line = ''.join(st.entry)
            l = line.split()

            if len(l) >= 10:
                if l[0] != 'Gen':
                    g = int(l[0])
                    i = l[1] # int(l[1])
                    e = float(l[3])
                    d = float(l[5])
                    s = l[6]
                    if s == 'N/A' or float(s) >= 0:
                        continue
                    if g in gene:
                        gene[g].append(d)
                        id_g[g].append(i)
                        dens_g[g].append(d)
                        op_g[g].append(l[2])
                        energy_g[g].append(e)
                    else:
                        gene[g] = [d]
                        id_g[g] = [i]
                        dens_g[g] = [d]
                        op_g[g] = [l[2]]
                        energy_g[g] = [e]
        st.close()
    # if n==-1:
    ng  = len(gene)
    # else:
    #    ng  = str(n)
    # x   = np.array(dens_g[ng])
    return ng,id_g,dens_g,energy_g,op_g
