#!/usr/bin/env python
# coding: utf-8
import networkx as nx
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.irff_np import IRFF_NP
from irff.structures import structure
from ase.visualize import view
import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np
import argh
import argparse
# get_ipython().run_line_magic('matplotlib', 'inline')


def nxp(gen='C2H4.gen',ffield='ffield.json',nn='T',threshold=0.1):
    atoms = read(gen)
    atom_name = atoms.get_chemical_symbols()
    nn_=True if nn=='T' else False

    ir = IRFF_NP(atoms=atoms,
                libfile=ffield,
                rcut=None,
                nn=nn_)
    # ir.get_pot_energy(atoms)
    # ir.logout()
    ir.calculate_Delta(atoms)
    natom = ir.natom

    g = nx.Graph()
    g.clear()
    color = {'C':'grey','H':'yellow','O':'red','N':'blue'}
    size = {'C':400,'H':150,'O':300,'N':300}
    nodeColor = []
    nodeSize  = []
    labels0,labels1 = {},{}
    for i in range(natom):
        c = color[atom_name[i]]
        s = size[atom_name[i]]
        nodeColor.append(c)
        nodeSize.append(s)
        g.add_node(atom_name[i]+str(i))
        labels0[atom_name[i]+str(i)] = atom_name[i]+str(i) + ':%4.3f' %ir.Deltap[i]
        labels1[atom_name[i]+str(i)] = atom_name[i]+str(i) + ':%4.3f' %ir.Delta[i]

        
    edgew = []
    for i in range(natom-1):
        for j in range(i+1,natom):
            if ir.r[i][j]<ir.r_cut[i][j]:
               if ir.bop[i][j]>=threshold:
                  g.add_edge(atom_name[i]+str(i),atom_name[j]+str(j),
                             BO0='%5.4f' %ir.bop[i][j])
                  edgew.append(2.0*ir.bop[i][j])
            
    pos = {} # pos = nx.spring_layout(g)
    for i,a in enumerate(atoms):
        pos[atom_name[i]+str(i)] = [a.x,a.y]
        
    nx.draw(g,pos,node_color=nodeColor,node_size=nodeSize,width=edgew,with_labels=False)
    edge_labels = nx.get_edge_attributes(g, 'BO0')
    nx.draw_networkx_edge_labels(g,pos,labels=edge_labels,font_size=8)
    # nx.draw_networkx_labels(g,pos,labels=labels0,font_size=8)

    plt.savefig('%s_bo0.eps' %gen.split('.')[0])
    plt.close()

    g = nx.Graph()
    g.clear()
    for i in range(natom):
        g.add_node(atom_name[i]+str(i))

    edgew = []
    for i in range(natom-1):
        for j in range(i+1,natom):
            if ir.r[i][j]<ir.r_cut[i][j]:
               if ir.bo0[i][j]>=threshold:
                  g.add_edge(atom_name[i]+str(i),atom_name[j]+str(j),
                             BO1='%5.4f' %ir.bo0[i][j])
                  edgew.append(2.0*ir.bo0[i][j])

    nx.draw(g,pos,node_color=nodeColor,node_size=nodeSize,width=edgew,with_labels=False)
    edge_labels = nx.get_edge_attributes(g, 'BO1')
    nx.draw_networkx_edge_labels(g,pos,labels=edge_labels,font_size=8)
    # nx.draw_networkx_labels(g,pos,labels=labels1,font_size=8)

    plt.savefig('%s_bo1.eps' %gen.split('.')[0])
    plt.close()
    ir.close()


if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [nxp])
   argh.dispatch(parser)
