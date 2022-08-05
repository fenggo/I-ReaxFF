#!/usr/bin/env python
# -*- coding: utf-8 -*-
from irff.dft.SinglePointEnergy import SinglePointEnergies
from ase.io import read


if __name__ == '__main__':
   ''''
   *.traj: contains the structure to be calculated by DFT
   label : the trajectory name include the energies and sturcutres calculated by DFT
   frame : number of frames to be collected to calculate the energy, if frame> the frame of *.traj contains then
           equal the number of frame of *.traj contains
   cpu   : number of cpu to be parallelly used 
   '''
   SinglePointEnergies('swing.traj',label='nm2-s',frame=10,cpu=4)




