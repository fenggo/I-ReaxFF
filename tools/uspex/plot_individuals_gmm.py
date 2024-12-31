#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from sklearn import svm


class Stack():
    def __init__(self,entry=[]):
        self.entry = entry
        
    def push(self,x):
        self.entry.append(x) 

    def pop(self):
        return self.entry.pop()
    
    def close(self):
        self.entry = None

#colors = ["navy", "turquoise", "darkorange"]
colors  = ['#1d9bf7','#c65861','#ffa725','#be588d','#35a153','#f26a11'] #,'#444577'
n_classes = 6

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        # print(covariances)

        v, w = np.linalg.eigh(covariances)
        # print('\nv\n',v,'\n w \n',w)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], 1.5*v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_linestyle('dashed')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        # ax.set_aspect("equal", "datalim")

def plot_indiv(findi='Individuals'):
    gene      = {}
    density   = []
    id_g      = {}
    dens_g    = {}
    op_g      = {}

    with open(findi) as f:
         for line in f.readlines():
             st = Stack([])
             for x in line:
                if x!=']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ !='[':
                        x_ = st.pop()
             line = ''.join(st.entry)
             l = line.split()
             
             if len(l)>=10:
                if l[0] != 'Gen':
                   g = l[0]
                   i = int(l[1])
                   e = float(l[3]) + 90
                   d = float(l[5])
                   s = l[6]
                   if s=='N/A' or float(s)>=0:
                      continue
                   if g in gene:  
                      gene[g].append(d)
                      id_g[g].append(i)
                      dens_g[g].append([d,e])
                      op_g[g].append(l[2])
                   else:
                      gene[g]    = [d]
                      id_g[g]    = [i]
                      dens_g[g]  = [[d,e]]
                      op_g[g]    = [l[2]]
         st.close()
    
    ng      = str(len(gene))

    density = {}
    enthalpy= {}
    #cluster = {}

    X         = np.array(dens_g[ng])
    X       = np.array(dens_g[ng])
    # Try GMMs using different types of covariances.
    
    for i,op in enumerate(op_g[ng]):
        if op not in density:
           density[op]  = [dens_g[ng][i][0]]
           enthalpy[op] = [dens_g[ng][i][1]]
        else:
           density[op].append(dens_g[ng][i][0])
           enthalpy[op].append(dens_g[ng][i][1])

    plt.figure()
    plt.ylabel(r'$Enthalpy$ ($eV$)',fontdict={'size':10})
    plt.xlabel(r'$Density$ ($g/cm^3$)',fontdict={'size':10})

    ax = plt.subplot()

    markers = {'Heredity':'o','keptBest':'s','softmutate':'^',
               'Rotate':'v','Permutate':'p','Random':'d'}
    colors_dic  = {'Heredity':'#1d9bf7','keptBest':'#ffbd15','softmutate':'#fcff07',
                   'Rotate':'#fe5357','Permutate':'#35a153','Random':'#303cf9'}
    
    
    for op in density:
        mk = markers[op]
        ax.scatter(density[op],enthalpy[op],alpha=0.9,
                    marker=mk,color=colors_dic[op],
                    edgecolor=colors_dic[op],s=30,
                    label=op)
    # cov_type = ["spherical", "diag", "tied", "full"]
    gmm = GaussianMixture(n_components=n_classes, 
                          covariance_type='full', max_iter=20, 
                          random_state=0)

    # Train the other parameters using the EM algorithm.
    gmm.fit(X)
    cla = gmm.predict(X)
    # X_,cla_ = gmm.sample(1000)
    # ax.scatter(X_[:,0],X_[:,1],alpha=0.9,color='k',s=1)
    make_ellipses(gmm, ax)
    
    classes = {i:[] for i in range(n_classes)}
    for i,c in enumerate(cla):
        classes[c].append(id_g[ng][i])

    for i in range(n_classes):
        print('\n- {:d} -\n'.format(i),classes[i])
    
    # ax.set_xlim(1.4,2.0)
    # plt.xticks(())
    # plt.yticks(())
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    # plt.show()
    plt.savefig('individuals_kmean.pdf',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''
   plot_indiv(findi='Individuals')

